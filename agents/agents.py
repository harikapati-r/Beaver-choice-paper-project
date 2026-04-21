"""
Pydantic-AI agent definitions for Beaver's Choice Paper Company.

Agents are created lazily (on first access) so that importing this module
does not require OPENAI_API_KEY to be set at import time.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from agents.tools import (
    inventory_check_tool,
    inventory_overview_tool,
    reorder_assessment_tool,
    process_reorder_tool,
    quote_history_tool,
    price_calculator_tool,
    quote_generator_tool,
    sales_feasibility_tool,
    delivery_schedule_tool,
    process_sale_tool,
    financial_report_tool,
    cash_balance_tool,
)


@lru_cache(maxsize=1)
def _model() -> OpenAIModel:
    return OpenAIModel("gpt-4o-mini")


@lru_cache(maxsize=1)
def get_inventory_agent() -> Agent:
    return Agent(
        _model(),
        system_prompt=(
            "You are the Inventory Agent for Beaver's Choice Paper Company. "
            "Check stock levels, assess reorder needs, and manage stock operations "
            "using the provided tools. Always call the appropriate tool before answering."
        ),
        tools=[inventory_check_tool, inventory_overview_tool,
               reorder_assessment_tool, process_reorder_tool],
        retries=2,
    )


@lru_cache(maxsize=1)
def get_quoting_agent() -> Agent:
    return Agent(
        _model(),
        system_prompt=(
            "You are the Quoting Agent for Beaver's Choice Paper Company. "
            "Generate competitive quotes with bulk discounts and research historical pricing. "
            "Always call the appropriate tool before answering."
        ),
        tools=[quote_history_tool, price_calculator_tool, quote_generator_tool],
        retries=2,
    )


@lru_cache(maxsize=1)
def get_sales_agent() -> Agent:
    return Agent(
        _model(),
        system_prompt=(
            "You are the Sales Agent for Beaver's Choice Paper Company. "
            "Check feasibility, schedule deliveries, and finalize sale transactions "
            "using the provided tools. Always call the appropriate tool before answering."
        ),
        tools=[sales_feasibility_tool, delivery_schedule_tool, process_sale_tool],
        retries=2,
    )


@lru_cache(maxsize=1)
def get_financial_agent() -> Agent:
    return Agent(
        _model(),
        system_prompt=(
            "You are the Financial Agent for Beaver's Choice Paper Company. "
            "Generate financial reports and monitor cash balance using the provided tools. "
            "Always call the appropriate tool before answering."
        ),
        tools=[financial_report_tool, cash_balance_tool],
        retries=2,
    )


# Convenience aliases kept for backwards-compatibility with any direct imports
inventory_agent = property(get_inventory_agent)
quoting_agent = property(get_quoting_agent)
sales_agent = property(get_sales_agent)
financial_agent = property(get_financial_agent)
