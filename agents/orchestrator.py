"""Orchestrator: coordinates the four specialist agents to handle customer requests."""
import json
import re
from datetime import datetime
from typing import Dict, List

from agents.db import paper_supplies
from agents.agents import (
    get_inventory_agent,
    get_quoting_agent,
    get_sales_agent,
    get_financial_agent,
)


# ---------------------------------------------------------------------------
# Request parsing (R7 fix: deduplicate and filter zero-quantity results)
# ---------------------------------------------------------------------------

def parse_customer_request(request: str) -> Dict:
    """
    Extract line items from a natural-language customer request.

    Deduplicates matches so the same catalog item is counted only once
    (using the highest quantity if the same item is matched by multiple patterns).
    Filters out any zero-quantity results.
    """
    # Stop each capture at comma, newline, or " and " so multi-item sentences parse correctly.
    _STOP = r"(?:,|\band\b|$|\n)"
    patterns = [
        r"(\d+)\s+sheets?\s+of\s+((?:(?!\band\b)[^,\n])+)",
        r"(\d+)\s+((?:(?!\band\b)[^,\n])*paper(?:(?!\band\b)[^,\n])*)",
        r"(\d+)\s+((?:(?!\band\b)[^,\n])*cardstock(?:(?!\band\b)[^,\n])*)",
        r"(\d+)\s+((?:(?!\band\b)[^,\n])*envelope(?:(?!\band\b)[^,\n])*)",
        r"(\d+)\s+((?:(?!\band\b)[^,\n])*plate(?:(?!\band\b)[^,\n])*)",
        r"(\d+)\s+((?:(?!\band\b)[^,\n])*cup(?:(?!\band\b)[^,\n])*)",
        r"(\d+)\s+((?:(?!\band\b)[^,\n])*napkin(?:(?!\band\b)[^,\n])*)",
        r"(\d+)\s+roll[s]?\s+of\s+((?:(?!\band\b)[^,\n])+)",
        r"(\d+)\s+pack[s]?\s+of\s+((?:(?!\band\b)[^,\n])+)",
        r"(\d+)\s+ream[s]?\s+of\s+((?:(?!\band\b)[^,\n])+)",
    ]

    # best_match: {item_name -> max quantity seen}
    best_match: Dict[str, int] = {}

    for pattern in patterns:
        for match in re.findall(pattern, request, re.IGNORECASE):
            try:
                quantity = int(match[0])
            except ValueError:
                continue
            if quantity <= 0:
                continue

            item_description = match[1].strip()
            desc_words = set(item_description.lower().split())

            top_name, top_score = None, 0
            for supply in paper_supplies:
                score = len(set(supply["item_name"].lower().split()) & desc_words)
                if score > top_score:
                    top_score = score
                    top_name = supply["item_name"]

            if top_name and top_score > 0:
                # Keep the highest quantity if the same catalog item is matched twice
                if top_name not in best_match or quantity > best_match[top_name]:
                    best_match[top_name] = quantity

    items = [{"item_name": name, "quantity": qty} for name, qty in best_match.items()]
    total_items = sum(i["quantity"] for i in items)

    if total_items > 5000:
        order_size = "large"
    elif total_items > 1000:
        order_size = "medium"
    else:
        order_size = "small"

    return {"items": items, "order_size": order_size, "total_items": total_items}


# ---------------------------------------------------------------------------
# Orchestrator  (R3 fix: agents are actually invoked via run_sync)
# ---------------------------------------------------------------------------

def call_multi_agent_system(customer_request: str, request_date: str = None) -> str:
    """
    Coordinate the four specialist agents to fulfil a customer request.

    Each agent is invoked with run_sync() so it reasons through the task,
    calls its own tools, and returns a structured result.
    """
    if request_date is None:
        request_date = datetime.now().strftime("%Y-%m-%d")

    parsed = parse_customer_request(customer_request)
    if not parsed["items"]:
        return (
            "I apologize, but I couldn't identify specific paper products in your request. "
            "Please specify the items and quantities you need."
        )

    items_json = json.dumps(parsed["items"])
    order_size = parsed["order_size"]

    # ------------------------------------------------------------------
    # Step 1 — Inventory Agent: check stock for every requested item
    # ------------------------------------------------------------------
    inventory_prompt = (
        f"Today is {request_date}. A customer wants to order the following items: {items_json}. "
        "For each item call inventory_check_tool to get the current stock level. "
        "Return a JSON list where each entry has: item_name, requested_quantity, available_stock, feasible (bool)."
    )
    inv_result = get_inventory_agent().run_sync(inventory_prompt)
    inv_text = inv_result.output if hasattr(inv_result, "output") else str(inv_result)

    # ------------------------------------------------------------------
    # Step 2 — Quoting Agent: generate a priced quote
    # ------------------------------------------------------------------
    quote_prompt = (
        f"Generate a quote for the following {order_size} order placed on {request_date}: {items_json}. "
        "Call quote_generator_tool with the items list and order_size, then return the full quote_explanation "
        "and a JSON list of items with their final_price."
    )
    quote_result = get_quoting_agent().run_sync(quote_prompt)
    quote_text = quote_result.output if hasattr(quote_result, "output") else str(quote_result)

    # ------------------------------------------------------------------
    # Step 3 — Sales Agent: check feasibility, schedule delivery,
    #           and commit transactions for available items
    # ------------------------------------------------------------------
    sales_prompt = (
        f"Date: {request_date}. Order items: {items_json} (order_size={order_size}). "
        "1) Call sales_feasibility_tool to determine which items can be fulfilled from stock. "
        "2) For items that ARE feasible, call delivery_schedule_tool to get estimated delivery dates. "
        "3) Call process_sale_tool for the feasible items (pass them with their final_price from the quote: "
        f"{quote_text}). "
        "Return: feasible_items list, unavailable_items list, delivery date, and total_revenue."
    )
    sales_result = get_sales_agent().run_sync(sales_prompt)
    sales_text = sales_result.output if hasattr(sales_result, "output") else str(sales_result)

    # ------------------------------------------------------------------
    # Step 4 — Inventory Agent (reorders): arrange restocking for
    #           any items that were out of stock
    # ------------------------------------------------------------------
    reorder_prompt = (
        f"Date: {request_date}. The following items could not be fulfilled from current stock: "
        f"(see sales result: {sales_text}). "
        "For each unfulfilled item call process_reorder_tool to arrange restocking. "
        "Return a list of items with their expected restock date."
    )
    reorder_result = get_inventory_agent().run_sync(reorder_prompt)
    reorder_text = reorder_result.output if hasattr(reorder_result, "output") else str(reorder_result)

    # ------------------------------------------------------------------
    # Step 5 — Financial Agent: confirm updated cash balance
    # ------------------------------------------------------------------
    financial_prompt = (
        f"As of {request_date}, call cash_balance_tool and return the current cash balance."
    )
    fin_result = get_financial_agent().run_sync(financial_prompt)
    fin_text = fin_result.output if hasattr(fin_result, "output") else str(fin_result)

    # ------------------------------------------------------------------
    # Compose the customer-facing response
    # ------------------------------------------------------------------
    response = (
        f"Thank you for your interest in Beaver's Choice Paper Company!\n\n"
        f"**Quote:**\n{quote_text}\n\n"
        f"**Order Status:**\n{sales_text}\n\n"
        f"**Restock Updates:**\n{reorder_text}\n\n"
        f"**Current Cash Balance:** {fin_text}"
    )
    return response
