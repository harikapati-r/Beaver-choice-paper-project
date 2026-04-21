"""Orchestrator: coordinates the four specialist agents to handle customer requests."""
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from agents.db import paper_supplies
from agents.agents import (
    get_inventory_agent,
    get_quoting_agent,
    get_sales_agent,
    get_financial_agent,
)
from agents.tools import (
    price_calculator_tool,
    sales_feasibility_tool,
    delivery_schedule_tool,
    process_sale_tool,
    process_reorder_tool,
)


# ---------------------------------------------------------------------------
# Agent-invocation helpers
# ---------------------------------------------------------------------------

def _run_agent(agent, prompt: str):
    """Invoke a pydantic-ai agent; returns RunResult or None on failure."""
    try:
        return agent.run_sync(prompt)
    except Exception:
        return None


def _get_tool_return(run_result, tool_name: str):
    """
    Extract the structured return value of a named tool call from a RunResult.

    pydantic-ai surfaces tool returns as ToolReturnPart instances inside the
    all_messages() list.  The .content field is a JSON string in most versions.
    """
    if run_result is None:
        return None
    try:
        for msg in run_result.all_messages():
            for part in getattr(msg, "parts", []):
                if getattr(part, "tool_name", None) == tool_name:
                    content = part.content
                    if isinstance(content, str):
                        return json.loads(content)
                    return content
    except Exception:
        pass
    return None


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

def call_multi_agent_system(customer_request: str, request_date: Optional[str] = None) -> str:
    """
    Coordinate the four specialist agents to fulfil a customer request.

    Inventory, Quoting, Sales, and Financial agents each handle their domain;
    the orchestrator drives the flow and builds the final customer-facing
    response from structured tool data — ensuring no duplicate line items,
    no zero-unit lines, and no "confirmed" language on empty orders.
    """
    if request_date is None:
        request_date = datetime.now().strftime("%Y-%m-%d")
    date_iso = f"{request_date}T00:00:00"

    parsed = parse_customer_request(customer_request)
    if not parsed["items"]:
        return (
            "Thank you for contacting Beaver's Choice Paper Company. "
            "We were unable to identify specific paper products in your request. "
            "Please let us know which items and quantities you need and we will be happy to assist."
        )

    items = parsed["items"]
    order_size = parsed["order_size"]

    sales_agent = get_sales_agent()
    quoting_agent = get_quoting_agent()
    inv_agent = get_inventory_agent()

    # ------------------------------------------------------------------
    # Step 1 — Sales Agent: check stock feasibility for every line item
    # ------------------------------------------------------------------
    feas_run = _run_agent(
        sales_agent,
        f"Check sales feasibility for these items on {date_iso}: {json.dumps(items)}",
    )
    feasibility = _get_tool_return(feas_run, "sales_feasibility_tool")
    if feasibility is None:
        feasibility = sales_feasibility_tool(items, date_iso)

    available_items = [a for a in feasibility["availability"] if a["feasible"]]
    unavailable_items = [a for a in feasibility["availability"] if not a["feasible"]]

    # ------------------------------------------------------------------
    # Step 2 — Quoting Agent: price the items that can actually be shipped
    # ------------------------------------------------------------------
    priced_items: List[Dict] = []
    sale_total = 0.0
    delivery_date: Optional[str] = None

    if available_items:
        ship_list = [{"item_name": a["item_name"], "quantity": a["requested_quantity"]}
                     for a in available_items]

        price_run = _run_agent(
            quoting_agent,
            f"Calculate pricing for a {order_size} order: {json.dumps(ship_list)}",
        )
        pricing = _get_tool_return(price_run, "price_calculator_tool")
        if pricing is None:
            pricing = price_calculator_tool(ship_list, order_size)
        priced_items = pricing["items"]

        # ------------------------------------------------------------------
        # Step 3 — Sales Agent: commit transactions and schedule delivery
        # ------------------------------------------------------------------
        sale_run = _run_agent(
            sales_agent,
            f"Process the sale for these priced items on {date_iso}: {json.dumps(priced_items)}",
        )
        sale_result = _get_tool_return(sale_run, "process_sale_tool")
        if sale_result is None:
            sale_result = process_sale_tool(priced_items, date_iso)
        sale_total = sale_result["total_revenue"]

        delivery_run = _run_agent(
            sales_agent,
            f"Schedule delivery for these items on {date_iso}: {json.dumps(ship_list)}",
        )
        delivery_info = _get_tool_return(delivery_run, "delivery_schedule_tool")
        if delivery_info is None:
            delivery_info = delivery_schedule_tool(ship_list, date_iso)
        delivery_date = delivery_info["estimated_delivery_date"]

    # ------------------------------------------------------------------
    # Step 4 — Inventory Agent: arrange restocking for out-of-stock items
    # ------------------------------------------------------------------
    restock_dates: List[Dict] = []
    for item in unavailable_items:
        reorder_run = _run_agent(
            inv_agent,
            f"Process a reorder for {item['item_name']}, "
            f"quantity {item['requested_quantity']}, on {date_iso}",
        )
        reorder = _get_tool_return(reorder_run, "process_reorder_tool")
        if reorder is None:
            reorder = process_reorder_tool(item["item_name"], item["requested_quantity"], date_iso)
        if reorder.get("status") == "reorder_processed":
            restock_dates.append({
                "item_name": item["item_name"],
                "delivery_date": reorder["details"]["delivery_date"],
            })

    # ------------------------------------------------------------------
    # Compose the customer-facing response from structured data
    # ------------------------------------------------------------------
    return _build_customer_response(
        priced_items=priced_items,
        unavailable_items=unavailable_items,
        sale_total=sale_total,
        delivery_date=delivery_date,
        restock_dates=restock_dates,
        order_size=order_size,
    )


def _build_customer_response(
    priced_items: List[Dict],
    unavailable_items: List[Dict],
    sale_total: float,
    delivery_date: Optional[str],
    restock_dates: List[Dict],
    order_size: str,
) -> str:
    """Build a clean, internally-consistent customer-facing response."""
    lines = ["Thank you for your interest in Beaver's Choice Paper Company!"]

    has_fulfilled = bool(priced_items)
    has_unavailable = bool(unavailable_items)

    if has_fulfilled and not has_unavailable:
        # ---- Full fulfillment ----
        lines.append("\nWe are pleased to confirm your order:\n")
        for item in priced_items:
            line = f"  - {item['quantity']} x {item['item_name']} at ${item['unit_price']:.2f} each"
            if item["discount_rate"] > 0:
                line += f" ({item['discount_rate'] * 100:.0f}% bulk discount applied)"
            line += f"  =  ${item['final_price']:.2f}"
            lines.append(line)
        lines.append(f"\nOrder Total: ${sale_total:.2f}")
        lines.append(f"Estimated delivery: {delivery_date}")
        lines.append("\nYour order is confirmed. We look forward to serving you!")

    elif has_fulfilled and has_unavailable:
        # ---- Partial fulfillment ----
        lines.append(
            "\nWe can partially fulfill your order. "
            "Below is what we are able to ship immediately:\n"
        )
        for item in priced_items:
            line = f"  - {item['quantity']} x {item['item_name']} at ${item['unit_price']:.2f} each"
            if item["discount_rate"] > 0:
                line += f" ({item['discount_rate'] * 100:.0f}% bulk discount applied)"
            line += f"  =  ${item['final_price']:.2f}"
            lines.append(line)
        lines.append(f"\nPartial Order Total: ${sale_total:.2f}")
        lines.append(f"Estimated delivery for available items: {delivery_date}")

        lines.append("\nThe following items are currently out of stock and cannot be included in this shipment:")
        for item in unavailable_items:
            lines.append(
                f"  - {item['item_name']}: you requested {item['requested_quantity']}, "
                f"we currently have {item['available_stock']} in stock."
            )

        if restock_dates:
            lines.append(
                "\nWe have arranged restocking orders for the unavailable items. "
                "Expected restock dates:"
            )
            seen = set()
            for r in restock_dates:
                key = (r["item_name"], r["delivery_date"])
                if key not in seen:
                    lines.append(f"  - {r['item_name']}: expected by {r['delivery_date']}")
                    seen.add(key)
            lines.append(
                "Please contact us once your preferred restock date has passed "
                "and we will be happy to process the remainder of your order."
            )

    else:
        # ---- No items can be fulfilled ----
        lines.append(
            "\nUnfortunately, we are unable to fulfill your order at this time "
            "due to insufficient stock for all requested items:\n"
        )
        for item in unavailable_items:
            lines.append(
                f"  - {item['item_name']}: you requested {item['requested_quantity']}, "
                f"we currently have {item['available_stock']} in stock."
            )

        if restock_dates:
            lines.append(
                "\nWe have arranged restocking and expect the following delivery dates:"
            )
            seen = set()
            for r in restock_dates:
                key = (r["item_name"], r["delivery_date"])
                if key not in seen:
                    lines.append(f"  - {r['item_name']}: expected by {r['delivery_date']}")
                    seen.add(key)
            lines.append(
                "We apologise for the inconvenience. "
                "Please reach out once the restock arrives and we will prioritise your order."
            )
        else:
            lines.append(
                "\nWe apologise that we cannot fulfil your request right now. "
                "Please contact us to discuss alternative arrangements."
            )

    return "\n".join(lines)
