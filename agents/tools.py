"""Business-logic tool functions used by pydantic-ai agents."""
from datetime import datetime
from typing import Dict, List, Optional

from agents.db import (
    paper_supplies,
    create_transaction,
    get_stock_level,
    get_all_inventory,
    get_supplier_delivery_date,
    get_cash_balance,
    generate_financial_report,
    search_quote_history,
)


# ---------------------------------------------------------------------------
# Inventory tools
# ---------------------------------------------------------------------------

def inventory_check_tool(item_name: str, date: Optional[str] = None) -> Dict:
    """Return current stock level for a specific item."""
    if date is None:
        date = datetime.now().isoformat()
    stock_info = get_stock_level(item_name, date)
    current_stock = int(stock_info["current_stock"].iloc[0]) if not stock_info.empty else 0
    item_details = next((i for i in paper_supplies if i["item_name"] == item_name), None)
    return {
        "item_name": item_name,
        "current_stock": current_stock,
        "status": "in_stock" if current_stock > 0 else "out_of_stock",
        "item_details": item_details,
    }


def inventory_overview_tool(date: Optional[str] = None) -> Dict:
    """Return a snapshot of all inventory with low-stock / adequate-stock split."""
    if date is None:
        date = datetime.now().isoformat()
    inventory = get_all_inventory(date)
    low_stock, adequate_stock = {}, {}
    for name, stock in inventory.items():
        (low_stock if int(stock) < 100 else adequate_stock)[name] = int(stock)
    return {"total_items": len(inventory), "low_stock_items": low_stock,
            "adequate_stock_items": adequate_stock, "date": date}


def reorder_assessment_tool(item_name: str, quantity_needed: int,
                             date: Optional[str] = None) -> Dict:
    """Decide whether a reorder is required and compute its cost/delivery date."""
    if date is None:
        date = datetime.now().isoformat()
    stock_info = get_stock_level(item_name, date)
    current_qty = int(stock_info["current_stock"].iloc[0]) if not stock_info.empty else 0
    if current_qty < quantity_needed:
        reorder_qty = max(quantity_needed - current_qty, 500)
        delivery_date = get_supplier_delivery_date(date, reorder_qty)
        item_details = next((i for i in paper_supplies if i["item_name"] == item_name), None)
        cost = float(reorder_qty * item_details["unit_price"]) if item_details else 0.0
        return {"needs_reorder": True, "current_stock": current_qty,
                "quantity_needed": int(quantity_needed), "reorder_quantity": reorder_qty,
                "delivery_date": delivery_date, "estimated_cost": cost}
    return {"needs_reorder": False, "current_stock": current_qty,
            "quantity_needed": int(quantity_needed)}


def process_reorder_tool(item_name: str, quantity: int,
                          date: Optional[str] = None) -> Dict:
    """Assess and (if needed) commit a reorder transaction."""
    info = reorder_assessment_tool(item_name, quantity, date)
    if info["needs_reorder"]:
        tid = create_transaction(item_name, "stock_orders",
                                 info["reorder_quantity"], info["estimated_cost"], date or datetime.now().isoformat())
        return {"status": "reorder_processed", "transaction_id": int(tid), "details": info}
    return {"status": "no_reorder_needed", "details": info}


# ---------------------------------------------------------------------------
# Quoting tools
# ---------------------------------------------------------------------------

def quote_history_tool(search_terms: List[str]) -> List[Dict]:
    """Search historical quotes for similar requests."""
    return search_quote_history(search_terms, limit=5)


def price_calculator_tool(items: List[Dict], order_size: str = "medium") -> Dict:
    """Calculate line-item pricing with bulk discounts."""
    total_cost = 0.0
    item_details = []
    for item in items:
        info = next((p for p in paper_supplies if p["item_name"] == item["item_name"]), None)
        if info:
            unit_price = info["unit_price"]
            qty = item["quantity"]
            subtotal = qty * unit_price
            if order_size == "large":
                rate = 0.15 if qty > 1000 else 0.10
            elif order_size == "medium":
                rate = 0.05 if qty > 500 else 0.03
            elif qty > 100:
                rate = 0.02
            else:
                rate = 0.0
            discount = subtotal * rate
            final_price = subtotal - discount
            item_details.append({"item_name": item["item_name"], "quantity": qty,
                                  "unit_price": unit_price, "subtotal": subtotal,
                                  "discount_rate": rate, "discount": discount,
                                  "final_price": final_price})
            total_cost += final_price
    return {"items": item_details, "total_cost": total_cost, "order_size": order_size}


def quote_generator_tool(customer_request: str, items: List[Dict], order_size: str) -> Dict:
    """Generate a complete quote with explanation text."""
    pricing = price_calculator_tool(items, order_size)
    explanation = f"Thank you for your {order_size} order request! "
    if pricing["total_cost"] > 500:
        explanation += "We've applied bulk discounts to provide you with the best value. "
    explanation += "Your order includes: "
    for item in pricing["items"]:
        explanation += f"{item['quantity']} {item['item_name']} at ${item['unit_price']:.2f} each"
        if item["discount_rate"] > 0:
            explanation += f" (with {item['discount_rate']*100:.0f}% bulk discount)"
        explanation += ", "
    explanation = explanation.rstrip(", ") + f". Total cost: ${pricing['total_cost']:.2f}"
    return {"total_amount": pricing["total_cost"], "quote_explanation": explanation,
            "items": pricing["items"], "order_size": order_size}


# ---------------------------------------------------------------------------
# Sales tools
# ---------------------------------------------------------------------------

def sales_feasibility_tool(items: List[Dict], date: Optional[str] = None) -> Dict:
    """Check whether current inventory can satisfy each line item."""
    if date is None:
        date = datetime.now().isoformat()
    feasible = True
    availability = []
    for item in items:
        stock_info = get_stock_level(item["item_name"], date)
        current_stock = int(stock_info["current_stock"].iloc[0]) if not stock_info.empty else 0
        item_feasible = current_stock >= item["quantity"]
        if not item_feasible:
            feasible = False
        availability.append({"item_name": item["item_name"],
                              "requested_quantity": item["quantity"],
                              "available_stock": current_stock,
                              "feasible": item_feasible})
    return {"feasible": feasible, "availability": availability, "date": date}


def delivery_schedule_tool(items: List[Dict], date: Optional[str] = None) -> Dict:
    """Calculate delivery dates for all items in an order."""
    if date is None:
        date = datetime.now().isoformat()
    delivery_details = []
    max_days = 0
    for item in items:
        delivery_date = get_supplier_delivery_date(date, item["quantity"])
        base = datetime.fromisoformat(date.split("T")[0])
        days = (datetime.fromisoformat(delivery_date) - base).days
        max_days = max(max_days, days)
        delivery_details.append({"item_name": item["item_name"], "quantity": item["quantity"],
                                  "delivery_date": delivery_date, "days_from_order": days})
    return {"estimated_delivery_date": delivery_details[0]["delivery_date"] if delivery_details else date,
            "max_delivery_days": max_days, "delivery_details": delivery_details}


def process_sale_tool(items: List[Dict], date: Optional[str] = None) -> Dict:
    """Commit sale transactions for all items and return totals."""
    if date is None:
        date = datetime.now().isoformat()
    total_revenue = 0.0
    transaction_ids = []
    for item in items:
        tid = create_transaction(item["item_name"], "sales",
                                 item["quantity"], item["final_price"], date)
        transaction_ids.append(int(tid))
        total_revenue += item["final_price"]
    return {"status": "sale_processed", "total_revenue": total_revenue,
            "transaction_ids": transaction_ids}


# ---------------------------------------------------------------------------
# Financial tools
# ---------------------------------------------------------------------------

def financial_report_tool(date: Optional[str] = None) -> Dict:
    if date is None:
        date = datetime.now().isoformat()
    return generate_financial_report(date)


def cash_balance_tool(date: Optional[str] = None) -> float:
    if date is None:
        date = datetime.now().isoformat()
    return get_cash_balance(date)
