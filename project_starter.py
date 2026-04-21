"""
Beaver's Choice Paper Company — multi-agent order processing system.

Entry point for evaluation. The business logic lives in the `agents/` package:
  agents/db.py          — catalog, database helpers
  agents/tools.py       — tool functions used by the agents
  agents/agents.py      — pydantic-ai Agent definitions
  agents/orchestrator.py — request parsing and agent coordination
"""
import time

import pandas as pd

from agents.db import init_database, generate_financial_report, db_engine
from agents.orchestrator import call_multi_agent_system

# Re-export symbols that callers / tests may import directly from this module
from agents.db import (
    paper_supplies,
    create_transaction,
    get_all_inventory,
    get_stock_level,
    get_supplier_delivery_date,
    get_cash_balance,
    search_quote_history,
)


def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"
        response = call_multi_agent_system(request_with_date, request_date)

        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append({
            "request_id": idx + 1,
            "request_date": request_date,
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": response,
        })

        time.sleep(1)

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    run_test_scenarios()
