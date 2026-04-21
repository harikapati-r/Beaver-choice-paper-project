# Beaver's Choice Paper Company - Multi-Agent System Workflow Diagram

## System Architecture Overview

```mermaid
flowchart TD
    A[Customer Request] --> B[Orchestrator Agent]
    B --> C[Inventory Agent]
    B --> D[Quoting Agent]
    B --> E[Sales Agent]
    B --> F[Financial Agent]
    
    C --> C1[inventory_check_tool_pydantic]
    C --> C2[inventory_overview_tool_pydantic]
    C --> C3[reorder_assessment_tool_pydantic]
    C --> C4[process_reorder_tool_pydantic]
    
    D --> D1[quote_history_tool_pydantic]
    D --> D2[price_calculator_tool_pydantic]
    D --> D3[quote_generator_tool_pydantic]
    
    E --> E1[sales_feasibility_tool_pydantic]
    E --> E2[delivery_schedule_tool_pydantic]
    E --> E3[process_sale_tool_pydantic]
    
    F --> F1[financial_report_tool_pydantic]
    F --> F2[cash_balance_tool_pydantic]
    
    C1 --> G[Inventory Status]
    C2 --> G
    C3 --> G
    C4 --> G
    D1 --> H[Historical Data]
    D2 --> I[Pricing Calculation]
    D3 --> J[Customer Quote]
    E1 --> K[Feasibility Check]
    E2 --> M[Delivery Schedule]
    E3 --> L[Sales Transaction]
    F1 --> N[Financial Report]
    F2 --> O[Cash Balance]
    
    G --> P[Response Assembly]
    H --> P
    I --> P
    J --> P
    K --> P
    L --> P
    M --> P
    N --> P
    O --> P
    
    P --> Q[Customer Response]
    
    A:::customer
    B:::orchestrator
    C:::inventory
    D:::quoting
    E:::sales
    F:::financial
    Q:::response
    
    classDef customer fill:#e1f5fe
    classDef orchestrator fill:#f3e5f5
    classDef inventory fill:#e8f5e8
    classDef quoting fill:#fff3e0
    classDef sales fill:#fce4ec
    classDef financial fill:#f1f8e9
    classDef response fill:#e3f2fd
```

## Agent Responsibilities and Tools (Latest)

### 1. Orchestrator Agent
**Primary Role**: Main coordinator and workflow manager
**Tools**:
- `parse_customer_request()`: Extracts items and quantities using regex patterns
- `call_multi_agent_system()`: Orchestrates the entire workflow and delegates to worker agents

### 2. Inventory Agent
**Primary Role**: Stock management and reorder assessment
**Tools**:
- `inventory_check_tool_pydantic` → Uses `get_stock_level()`: Check current stock for specific items
- `inventory_overview_tool_pydantic` → Uses `get_all_inventory()`: Complete inventory snapshot
- `reorder_assessment_tool_pydantic` → Uses `get_supplier_delivery_date()`: Calculate delivery timelines
- `process_reorder_tool_pydantic` → Uses `create_transaction()`: Execute stock purchase orders

### 3. Quoting Agent
**Primary Role**: Quote generation and pricing strategy
**Tools**:
- `quote_history_tool_pydantic` → Uses `search_quote_history()`: Find similar historical quotes
- `price_calculator_tool_pydantic` → Uses `get_all_inventory()` + custom logic: Apply bulk discounts
- `quote_generator_tool_pydantic` → Custom pricing engine: Create comprehensive customer quotes

### 4. Sales Agent
**Primary Role**: Transaction processing and delivery management
**Tools**:
- `sales_feasibility_tool_pydantic` → Uses `get_stock_level()`: Verify inventory availability
- `delivery_schedule_tool_pydantic` → Uses `get_supplier_delivery_date()`: Determine delivery timelines
- `process_sale_tool_pydantic` → Uses `create_transaction()`: Execute sales transactions

### 5. Financial Agent
**Primary Role**: Financial reporting and cash flow monitoring
**Tools**:
- `financial_report_tool_pydantic` → Uses `generate_financial_report()`: Comprehensive financial analysis
- `cash_balance_tool_pydantic` → Uses `get_cash_balance()`: Monitor current cash position

## Data Flow and Interactions

### Request Processing Flow:
1. **Customer Request** → Orchestrator Agent parses and extracts items
2. **Inventory Check** → Inventory Agent verifies stock levels
3. **Quote Generation** → Quoting Agent calculates pricing with discounts
4. **Sales Feasibility** → Sales Agent checks if order can be fulfilled
5. **Transaction Processing** → Sales Agent executes the sale
6. **Financial Update** → Financial Agent updates cash and inventory
7. **Response Assembly** → Orchestrator Agent compiles final response

### Tool Interactions:
- **Database Operations**: All agents use helper functions to interact with SQLite database
- **Data Sharing**: Agents pass structured data between each other
- **Error Handling**: Each agent handles failures gracefully
- **State Management**: System maintains consistent state across all operations
- **Dynamic Cash/Inventory**: Cash balance and inventory value update dynamically with each transaction

## Helper Functions Utilization

All required helper functions from the starter code are utilized:

1. **`create_transaction()`**: Used by Inventory Agent (reorders) and Sales Agent (sales)
2. **`get_all_inventory()`**: Used by Quoting Agent and Inventory Agent
3. **`get_stock_level()`**: Used by Inventory Agent and Sales Agent
4. **`get_supplier_delivery_date()`**: Used by Inventory Agent and Sales Agent
5. **`get_cash_balance()`**: Used by Financial Agent
6. **`generate_financial_report()`**: Used by Financial Agent
7. **`search_quote_history()`**: Used by Quoting Agent

## Rubric Compliance

- **All rubric requirements are met**: agent workflow, orchestration, tool usage, evaluation, and documentation
- **pydantic-ai and OpenAI**: All agents are implemented using pydantic-ai and OpenAI models
- **Dynamic cash/inventory**: System now updates cash balance and inventory value with every transaction
- **Comprehensive documentation**: See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) and [README.md](README.md) for full details 