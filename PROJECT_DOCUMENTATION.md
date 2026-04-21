# Beaver's Choice Paper Company - Multi-Agent System Documentation

## Project Overview

This project implements a production-ready multi-agent system for the Beaver's Choice Paper Company to streamline inventory management, quoting, sales, and financial operations. The system consists of four specialized agents, orchestrated by a main function, and is fully compliant with all project rubric requirements.

## Agent Workflow Diagram

The system follows a hierarchical orchestration pattern with the following architecture:

```
Customer Request → Orchestrator Agent → Specialized Worker Agents → Response
```

- **Orchestrator Agent**: Main coordinator that parses customer requests and delegates tasks
- **Inventory Agent**: Manages stock levels, reorder assessments, and inventory reports
- **Quoting Agent**: Generates competitive quotes with bulk discounts
- **Sales Agent**: Finalizes transactions and manages delivery schedules
- **Financial Agent**: Provides financial reporting and cash flow monitoring

All agents are implemented using the [pydantic-ai](https://github.com/ContextualAI/pydantic-ai) framework and use OpenAI models for reasoning and tool selection. The orchestrator agent coordinates all workflow and delegates tasks to the worker agents.

For the complete workflow diagram and agent architecture, see [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md).

## Framework Selection and Justification

**pydantic-ai Framework**: This implementation uses the recommended pydantic-ai framework for the following reasons:

1. **Direct Integration**: Seamless integration with the provided helper functions
2. **Performance Optimization**: Efficient agent orchestration and tool selection
3. **Maintainability**: Clear, readable code with modular agent definitions
4. **Scalability**: Modular design allows easy addition of new agents
5. **Industry Alignment**: Leverages OpenAI models for advanced reasoning

## Implementation Details

### Agent Classes

#### 1. Orchestrator Agent
- **Primary Role**: Coordinates all other agents and processes customer requests
- **Key Methods**:
  - `parse_customer_request()`: Extracts items and quantities from customer text
  - `call_multi_agent_system()`: Main workflow orchestrator

#### 2. Inventory Agent
- **Primary Role**: Inventory management and stock assessments
- **Key Tools**:
  - `inventory_check_tool_pydantic()`: Check current stock levels (uses `get_stock_level()`)
  - `inventory_overview_tool_pydantic()`: Complete inventory status (uses `get_all_inventory()`)
  - `reorder_assessment_tool_pydantic()`: Determine if reordering is necessary (uses `get_supplier_delivery_date()`)
  - `process_reorder_tool_pydantic()`: Execute reorder transactions (uses `create_transaction()`)

#### 3. Quoting Agent
- **Primary Role**: Quote generation and pricing
- **Key Tools**:
  - `quote_history_tool_pydantic()`: Find similar historical quotes (uses `search_quote_history()`)
  - `price_calculator_tool_pydantic()`: Apply bulk discounts and calculate costs (uses `get_all_inventory()` + custom logic)
  - `quote_generator_tool_pydantic()`: Create comprehensive customer quotes

#### 4. Sales Agent
- **Primary Role**: Transaction processing and delivery management
- **Key Tools**:
  - `sales_feasibility_tool_pydantic()`: Verify inventory availability (uses `get_stock_level()`)
  - `delivery_schedule_tool_pydantic()`: Determine delivery timelines (uses `get_supplier_delivery_date()`)
  - `process_sale_tool_pydantic()`: Execute sales transactions (uses `create_transaction()`)

#### 5. Financial Agent
- **Primary Role**: Financial reporting and monitoring
- **Key Tools**:
  - `financial_report_tool_pydantic()`: Generate comprehensive financial reports (uses `generate_financial_report()`)
  - `cash_balance_tool_pydantic()`: Monitor current cash position (uses `get_cash_balance()`)

### Tools and Helper Functions

The system utilizes all required helper functions from the starter code:

1. **create_transaction()**: Records sales and stock orders
2. **get_all_inventory()**: Retrieves complete inventory snapshot
3. **get_stock_level()**: Checks specific item stock levels
4. **get_supplier_delivery_date()**: Calculates delivery timelines
5. **get_cash_balance()**: Monitors cash flow
6. **generate_financial_report()**: Creates financial summaries
7. **search_quote_history()**: Finds relevant historical quotes

### Request Parsing Logic

The system uses regex patterns to extract items and quantities from customer requests, ensuring robust parsing for a variety of request formats.

### Pricing Strategy

The system implements intelligent bulk discounting:
- **Large orders** (>5000 items): 10-15% discount
- **Medium orders** (>1000 items): 3-5% discount
- **Small orders** (>100 items): 2% discount

## Evaluation Results (Latest)

The system was tested with 20 customer requests from `quote_requests_sample.csv`:

- ✅ **6 successful orders** completed
- 🔄 **12 partial fulfillments** with reorder arrangements
- ❌ **2 unfulfilled requests** due to parsing issues
- 💸 **Cash balance and inventory value now update dynamically with each transaction**

### Key Metrics (Latest Run)
- **Final Cash Balance**: $45,738.95
- **Final Inventory Value**: $3,592.90
- **Response Quality**: 90% success rate

### Rubric Compliance Verification

- **Agent Workflow Diagram**: Complete and included in `WORKFLOW_DIAGRAM.md`
- **Multi-Agent System Implementation**: All agents and tools implemented using pydantic-ai and OpenAI
- **Evaluation and Reflection**: All requirements met, with dynamic cash/inventory updates and clear customer-facing outputs
- **Industry Best Practices**: Readable, modular, and well-documented code; no sensitive information exposed

## Evaluation and Reflection

### Dataset and Evaluation Methodology

The multi-agent system was evaluated using the full set of customer requests provided in `quote_requests_sample.csv`. The results of this evaluation are documented in `test_results.csv`, which records the cash balance, inventory value, and system response after each request.

### Rubric Demonstration

- **At least three requests result in a change to the cash balance:**
  - Request 1: Cash balance increases from $45,059.70 to $45,129.30 after a successful order (see test_results.csv row 1).
  - Request 4: Cash increases to $45,128.80 after another successful order (row 4).
  - Request 5: Cash decreases to $45,038.80 after a partial fulfillment (row 5).
  - Many subsequent requests also result in cash balance changes, as seen throughout the file.

- **At least three quote requests are successfully fulfilled:**
  - Request 1: "Order confirmed! Total: $69.60" (row 1)
  - Request 4: "Order confirmed! Total: $24.50" (row 4)
  - Request 10: "Order confirmed! Total: $291.00" (row 10)
  - Request 11, 13, and 14 are also fully fulfilled, exceeding the requirement.

- **Not all requests are fulfilled, with reasons provided or implied:**
  - Request 2: "Unfortunately, we cannot fulfill your complete order due to insufficient inventory. None of the requested items are currently available in sufficient quantities."
  - Request 3, 5, 6, 7, 8, 9, 12, 16, 17, 18: Partial fulfillments with explanations and restock information.
  - Request 20: "I apologize, but I couldn't identify specific paper products in your request. Please specify the items and quantities you need."

### Reflection on Architecture and Implementation

The agent workflow diagram (see WORKFLOW_DIAGRAM.md) guided the design of a modular, orchestrator/worker agent system:
- **Orchestrator Agent**: Parses requests and delegates tasks to specialized agents.
- **Inventory Agent**: Handles stock checks and reordering.
- **Quoting Agent**: Generates quotes and applies pricing logic.
- **Sales Agent**: Processes sales and manages delivery.
- **Financial Agent**: Tracks cash and generates reports.

This separation of concerns ensures each agent has a clear, non-overlapping responsibility, making the system scalable and maintainable. The use of pydantic-ai and OpenAI models enables flexible, tool-driven reasoning for each agent.

### Evaluation Results and Strengths

- **Robust Transaction Handling**: The system updates cash balance and inventory value dynamically after each transaction, as shown in test_results.csv.
- **High Fulfillment Rate**: Over 6 requests are fully fulfilled, and most others are partially fulfilled with clear explanations.
- **Transparent Customer Communication**: Responses include pricing, discounts, delivery estimates, and reasons for unfulfilled or partial orders.
- **Graceful Degradation**: The system provides fallback responses for unrecognized or unfulfillable requests.
- **Comprehensive Use of Helper Functions**: All required starter code functions are used in agent tools.

### Suggestions for Further Improvement

1. **Enhanced Natural Language Parsing**: Integrate NLP models to better extract item and quantity information from complex or ambiguous customer requests, increasing the fulfillment rate.
2. **Predictive Inventory and Dynamic Pricing**: Use machine learning to forecast demand and optimize pricing based on historical data and market trends.
3. **Customer Relationship Features**: Track customer history and personalize offers or recommendations.
4. **Real-Time Supplier Integration**: Connect with external supplier APIs for live inventory and delivery updates.

---

## Reflection and Suggestions for Improvement

### Strengths Identified

1. **Robust Parsing**: Successfully identified items in 90% of requests
2. **Intelligent Discounting**: Applied appropriate bulk discounts based on order size
3. **Inventory Management**: Automatically triggered reorders for low stock items
4. **Customer Communication**: Provided clear explanations and delivery timelines
5. **Error Handling**: Gracefully handled parsing failures and inventory shortages

### Areas for Improvement

1. **Enhanced Parsing**: Implement NLP-based request understanding for better item recognition
2. **Dynamic Pricing**: Add machine learning for optimal pricing based on demand patterns
3. **Inventory Optimization**: Implement predictive analytics for stock management
4. **Customer Relationship Management**: Add customer history tracking and personalized offers
5. **Real-time Updates**: Implement webhook-based inventory updates
6. **Multi-language Support**: Add support for international customers
7. **Advanced Analytics**: Include business intelligence dashboards
8. **Integration APIs**: Connect with external suppliers and shipping providers

## Technical Architecture

### Database Integration

The system uses SQLite with the following tables:
- `transactions`: Records all sales and stock orders
- `inventory`: Current stock levels and item details
- `quote_requests`: Historical customer requests
- `quotes`: Previous quote data for reference

### Error Handling

The system includes comprehensive error handling:
- Invalid item parsing
- Insufficient inventory scenarios
- Database transaction failures
- Date parsing issues

### Scalability Considerations

The modular agent architecture allows for:
- Easy addition of new agent types
- Independent scaling of different functions
- Simple maintenance and debugging
- Clear separation of concerns

## Documentation & Reflection

- For the complete workflow diagram and agent architecture, see [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)
- For the rubric reflection and compliance, see the sections above and [README.md](README.md)

## Conclusion

The multi-agent system successfully demonstrates all rubric requirements:
- ✅ Handles customer inquiries efficiently
- ✅ Manages inventory with automatic reordering
- ✅ Generates competitive quotes with bulk discounts
- ✅ Processes sales transactions accurately
- ✅ Provides comprehensive financial reporting
- ✅ Cash balance and inventory value update dynamically

The system provides a solid foundation for the Beaver's Choice Paper Company's digital transformation, with clear paths for future enhancements and scalability. 

## Terminal Output Example

Below is a screenshot of the terminal output demonstrating the system's response to customer requests, showcasing the transparency and explainability of the outputs:

![Terminal Output](output_example.png)

The output includes:
- Customer request context and date
- Cash balance and inventory value before and after processing
- Detailed response with pricing, discounts, and delivery estimates
- Justifications for partial or unfulfilled orders

This ensures that all customer-facing interactions are transparent and provide necessary explanations without revealing sensitive internal information. 

## Reflection Report

### Architecture Explanation

The architecture of the multi-agent system is designed around a hierarchical orchestration pattern. The **Orchestrator Agent** acts as the main coordinator, delegating tasks to specialized worker agents:

- **Inventory Agent**: Manages stock levels and reordering.
- **Quoting Agent**: Generates quotes and applies pricing strategies.
- **Sales Agent**: Handles transactions and delivery schedules.
- **Financial Agent**: Provides financial reporting and cash flow monitoring.

This separation of concerns ensures scalability and maintainability, allowing each agent to focus on specific tasks without overlap.

### Evaluation Results

The system was tested with 20 customer requests, achieving:

- **6 successful orders**: Fully fulfilled requests.
- **12 partial fulfillments**: Orders partially fulfilled with arrangements for reordering.
- **2 unfulfilled requests**: Due to parsing issues.

Key metrics include a final cash balance of $45,738.95 and an inventory value of $3,592.90, with a 90% response quality success rate.

### Suggestions for Improvement

1. **Enhanced Natural Language Processing**: Integrate advanced NLP models to improve item and quantity extraction from complex customer requests, increasing fulfillment rates.

2. **Predictive Analytics for Inventory Management**: Implement machine learning models to forecast demand and optimize stock levels, reducing the likelihood of unfulfilled orders. 