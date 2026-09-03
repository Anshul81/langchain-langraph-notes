# Chapter 9.3: Custom Tools — Building Your Own with `@tool`

> **Phase 9 — Tools & Tool Calling** | [← Previous: Built-in Tools](chapter-31-builtin-tools.md) | [Next: Tool Calling Deep Dive →](chapter-33-tool-calling-deep-dive.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Create custom tools using the `@tool` decorator
- ✅ Use Pydantic models for structured tool input schemas
- ✅ Build tools that access APIs, databases, and external services
- ✅ Create tools with complex return types and error handling
- ✅ Understand `StructuredTool` and `BaseTool` for advanced use cases
- ✅ Build a **customer support toolkit** with multiple custom tools

| | |
|---|---|
| **Prerequisites** | Chapter 9.2 (Built-in Tools), Phase 6 (Chains & Runnables) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction

Built-in tools are great, but real-world applications need **custom tools** — tools that interact with *your* APIs, *your* databases, *your* business logic.

```
Built-in tools:  "Search the web", "Look up Wikipedia"    ← Generic
Custom tools:    "Look up customer order #12345"           ← YOUR business logic
                 "Check inventory for SKU-789"
                 "Create a support ticket in Jira"
                 "Query our internal knowledge base"
```

LangChain provides three ways to create custom tools, from simplest to most flexible:

| Method | When to Use |
|--------|-------------|
| **`@tool` decorator** | 90% of cases — simple functions |
| **`StructuredTool.from_function`** | When you need more config without a class |
| **`BaseTool` subclass** | Full control — async, callbacks, custom validation |

---

## Part 1: The `@tool` Decorator — Your Go-To Method

### Simplest Custom Tool

```python
from langchain_core.tools import tool

@tool
def get_word_count(text: str) -> int:
    """Count the number of words in a given text.
    
    Use this when the user asks about word count, text length, or 
    needs to know how many words are in a passage.
    """
    return len(text.split())


# Check what the LLM sees
print(f"Name:        {get_word_count.name}")
print(f"Description: {get_word_count.description}")
print(f"Args:        {get_word_count.args_schema.schema()}")
```

**Output:**
```
Name:        get_word_count
Description: Count the number of words in a given text.
    
    Use this when the user asks about word count, text length, or 
    needs to know how many words are in a passage.
Args:        {'properties': {'text': {'title': 'Text', 'type': 'string'}}, 
              'required': ['text'], 'title': 'get_word_countSchema', 'type': 'object'}
```

### How It Works

```
@tool decorator does three things:

1. NAME        ← taken from the function name
2. DESCRIPTION ← taken from the docstring (CRITICAL — LLM reads this!)
3. SCHEMA      ← inferred from type hints (str, int, float, etc.)

┌──────────────────────────────────────┐
│         @tool                        │
│                                      │
│  def my_function(arg: str) -> str:  │
│      """Description here."""         │
│           │            │       │     │
│           │            │       │     │
│      name = "my_function"      │     │
│                  │             │     │
│          schema from type hints      │
│                        │             │
│              description from docstring
└──────────────────────────────────────┘
```

### The Docstring Is Everything

The LLM decides whether to call your tool based **entirely** on the docstring. Make it count:

```python
# ❌ BAD — LLM has no idea when to use this
@tool
def lookup(id: str) -> str:
    """Looks up data."""
    ...

# ❌ BAD — Too technical, doesn't mention user intent
@tool
def lookup(id: str) -> str:
    """Executes a SELECT query against the orders PostgreSQL table 
    using the provided primary key with connection pooling via pgbouncer."""
    ...

# ✅ GOOD — Clear intent, clear inputs, clear use case
@tool
def lookup_order(order_id: str) -> str:
    """Look up a customer order by its order ID (e.g., 'ORD-12345').
    
    Use this when the user asks about:
    - Order status, tracking, or delivery
    - Order details (items, total, shipping address)
    - Order history for a specific order number
    
    Returns order details including status, items, and estimated delivery.
    """
    ...
```

---

## Part 2: Tools with Multiple Parameters

### Basic Multi-Parameter Tool

```python
@tool
def calculate_discount(
    original_price: float, 
    discount_percent: float, 
    tax_rate: float = 18.0
) -> str:
    """Calculate the final price after applying a discount and adding tax.
    
    Use when the user asks about discounts, sale prices, or final costs
    after applying a percentage discount.
    
    Args:
        original_price: The original price before discount (e.g., 1000.0)
        discount_percent: The discount percentage to apply (e.g., 20 for 20%)
        tax_rate: The tax rate as a percentage (default: 18% GST)
    """
    discounted = original_price * (1 - discount_percent / 100)
    final = discounted * (1 + tax_rate / 100)
    
    return (
        f"Original: ₹{original_price:,.2f}\n"
        f"Discount: {discount_percent}% (-₹{original_price * discount_percent / 100:,.2f})\n"
        f"After discount: ₹{discounted:,.2f}\n"
        f"Tax: {tax_rate}% (+₹{discounted * tax_rate / 100:,.2f})\n"
        f"Final price: ₹{final:,.2f}"
    )


# Test directly
print(calculate_discount.invoke({
    "original_price": 5000, 
    "discount_percent": 20
}))
```

**Output:**
```
Original: ₹5,000.00
Discount: 20% (-₹1,000.00)
After discount: ₹4,000.00
Tax: 18% (+₹720.00)
Final price: ₹4,720.00
```

### ⚠️ The `Args` Docstring Section Matters

```python
# LLM sees parameter descriptions from the Args section of the docstring
# If you don't write Args, the LLM only sees the type hints — often not enough

print(calculate_discount.args_schema.schema())
```

```json
{
  "properties": {
    "original_price": {
      "description": "The original price before discount (e.g., 1000.0)",
      "title": "Original Price",
      "type": "number"
    },
    "discount_percent": {
      "description": "The discount percentage to apply (e.g., 20 for 20%)",
      "title": "Discount Percent",
      "type": "number"
    },
    "tax_rate": {
      "description": "The tax rate as a percentage (default: 18% GST)",
      "default": 18.0,
      "title": "Tax Rate",
      "type": "number"
    }
  },
  "required": ["original_price", "discount_percent"]
}
```

---

## Part 3: Pydantic Input Schemas — Maximum Control

For complex tools, use Pydantic models to define the input schema with validation, descriptions, and constraints:

### Basic Pydantic Schema

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class SearchTicketsInput(BaseModel):
    """Input for searching support tickets."""
    
    query: str = Field(
        description="Search query — keywords to look for in ticket title and description"
    )
    status: str = Field(
        default="all",
        description="Filter by status: 'open', 'in_progress', 'resolved', 'all'"
    )
    priority: str = Field(
        default="all",
        description="Filter by priority: 'low', 'medium', 'high', 'critical', 'all'"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of tickets to return (1-20)",
        ge=1,
        le=20
    )


@tool(args_schema=SearchTicketsInput)
def search_tickets(
    query: str, 
    status: str = "all", 
    priority: str = "all", 
    limit: int = 5
) -> str:
    """Search support tickets by keywords, status, and priority.
    
    Use when the user asks about support tickets, customer issues,
    or wants to find specific tickets.
    """
    # Simulated database query
    all_tickets = [
        {"id": "TK-001", "title": "Login page not loading", "status": "open", 
         "priority": "high", "customer": "Acme Corp"},
        {"id": "TK-002", "title": "Password reset not working", "status": "in_progress", 
         "priority": "medium", "customer": "TechStart"},
        {"id": "TK-003", "title": "Billing discrepancy in invoice", "status": "open", 
         "priority": "critical", "customer": "GlobalTech"},
        {"id": "TK-004", "title": "Feature request: dark mode", "status": "open", 
         "priority": "low", "customer": "DesignCo"},
        {"id": "TK-005", "title": "API rate limiting errors", "status": "resolved", 
         "priority": "high", "customer": "DataFlow"},
    ]
    
    # Filter
    results = all_tickets
    if status != "all":
        results = [t for t in results if t["status"] == status]
    if priority != "all":
        results = [t for t in results if t["priority"] == priority]
    
    # Simple keyword search
    if query:
        query_lower = query.lower()
        results = [t for t in results if query_lower in t["title"].lower()]
    
    results = results[:limit]
    
    if not results:
        return "No tickets found matching your criteria."
    
    output = f"Found {len(results)} ticket(s):\n\n"
    for t in results:
        output += f"📋 [{t['id']}] {t['title']}\n"
        output += f"   Status: {t['status']} | Priority: {t['priority']} | Customer: {t['customer']}\n\n"
    
    return output


# Check the schema the LLM sees
import json
print(json.dumps(search_tickets.args_schema.schema(), indent=2))
```

### Why Pydantic Is Better Than Raw Type Hints

```python
# With just type hints:
@tool
def search(query: str, limit: int = 5) -> str:
    """Search tickets."""
    ...
# Schema: {"query": {"type": "string"}, "limit": {"type": "integer"}}
# LLM doesn't know what query should contain or limit's range!

# With Pydantic:
# Schema includes: descriptions, defaults, min/max values, enums
# LLM knows EXACTLY what to pass
```

### Pydantic with Enums

```python
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CreateTicketInput(BaseModel):
    """Input for creating a support ticket."""
    
    title: str = Field(description="Short, descriptive title for the ticket")
    description: str = Field(description="Detailed description of the issue")
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Ticket priority level"
    )
    customer_email: str = Field(description="Customer's email address")


@tool(args_schema=CreateTicketInput)
def create_ticket(
    title: str, 
    description: str, 
    priority: Priority = Priority.MEDIUM,
    customer_email: str = ""
) -> str:
    """Create a new support ticket.
    
    Use when the user wants to create, file, or open a new support ticket
    or customer issue.
    """
    import uuid
    ticket_id = f"TK-{uuid.uuid4().hex[:6].upper()}"
    
    return (
        f"✅ Ticket created successfully!\n"
        f"   ID: {ticket_id}\n"
        f"   Title: {title}\n"
        f"   Priority: {priority.value}\n"
        f"   Customer: {customer_email}\n"
        f"   Status: open"
    )


# Test
print(create_ticket.invoke({
    "title": "Dashboard not loading",
    "description": "The analytics dashboard shows a blank page after login",
    "priority": "high",
    "customer_email": "user@example.com"
}))
```

---

## Part 4: Tools That Call External APIs

### Weather API Tool (Real-World Pattern)

```python
import json
import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """Input for weather lookup."""
    city: str = Field(description="City name, e.g., 'Mumbai', 'New York', 'London'")
    units: str = Field(
        default="metric",
        description="Temperature units: 'metric' (Celsius) or 'imperial' (Fahrenheit)"
    )


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "metric") -> str:
    """Get the current weather for a city.
    
    Use when the user asks about weather, temperature, or climate conditions
    for a specific location.
    """
    try:
        # Using wttr.in — free, no API key needed
        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=10,
            headers={"User-Agent": "LangChain-Tool"}
        )
        response.raise_for_status()
        data = response.json()
        
        current = data["current_condition"][0]
        temp_key = "temp_C" if units == "metric" else "temp_F"
        unit_symbol = "°C" if units == "metric" else "°F"
        
        return (
            f"Weather in {city}:\n"
            f"  🌡️ Temperature: {current[temp_key]}{unit_symbol}\n"
            f"  💧 Humidity: {current['humidity']}%\n"
            f"  🌤️ Condition: {current['weatherDesc'][0]['value']}\n"
            f"  💨 Wind: {current['windspeedKmph']} km/h ({current['winddir16Point']})\n"
            f"  👁️ Visibility: {current['visibility']} km"
        )
    except requests.exceptions.Timeout:
        return f"Error: Weather service timed out for {city}. Try again later."
    except requests.exceptions.HTTPError as e:
        return f"Error: Could not get weather for '{city}'. City may not exist. ({e})"
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"


# Test
print(get_weather.invoke({"city": "Mumbai"}))
```

### Database Query Tool

```python
@tool
def query_products(
    category: str = "all",
    min_price: float = 0,
    max_price: float = 999999,
    in_stock_only: bool = True
) -> str:
    """Search our product catalog by category and price range.
    
    Use when the user asks about products, pricing, inventory, or
    wants to find items in our store.
    
    Args:
        category: Product category — 'electronics', 'clothing', 'books', or 'all'
        min_price: Minimum price filter (default: 0)
        max_price: Maximum price filter (default: no limit)
        in_stock_only: If True, only show items currently in stock (default: True)
    """
    # Simulated product database
    products = [
        {"name": "Wireless Headphones", "category": "electronics", "price": 2499, "stock": 45},
        {"name": "Laptop Stand", "category": "electronics", "price": 1299, "stock": 12},
        {"name": "USB-C Hub", "category": "electronics", "price": 1899, "stock": 0},
        {"name": "Python T-Shirt", "category": "clothing", "price": 599, "stock": 100},
        {"name": "Developer Hoodie", "category": "clothing", "price": 1499, "stock": 25},
        {"name": "Clean Code (Book)", "category": "books", "price": 899, "stock": 8},
        {"name": "Design Patterns (Book)", "category": "books", "price": 1199, "stock": 0},
        {"name": "Mechanical Keyboard", "category": "electronics", "price": 3999, "stock": 7},
    ]
    
    # Apply filters
    results = products
    if category != "all":
        results = [p for p in results if p["category"] == category]
    results = [p for p in results if min_price <= p["price"] <= max_price]
    if in_stock_only:
        results = [p for p in results if p["stock"] > 0]
    
    if not results:
        return "No products found matching your criteria."
    
    output = f"Found {len(results)} product(s):\n\n"
    for p in results:
        stock_status = f"✅ {p['stock']} in stock" if p["stock"] > 0 else "❌ Out of stock"
        output += f"  🛍️ {p['name']}\n"
        output += f"     ₹{p['price']:,} | {p['category']} | {stock_status}\n\n"
    
    return output


# Test
print(query_products.invoke({"category": "electronics", "max_price": 2000}))
```

---

## Part 5: `StructuredTool.from_function` — Middle Ground

When you want more configuration than `@tool` but don't want to write a class:

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CurrencyInput(BaseModel):
    amount: float = Field(description="The amount to convert")
    from_currency: str = Field(description="Source currency code, e.g., 'USD', 'INR', 'EUR'")
    to_currency: str = Field(description="Target currency code, e.g., 'USD', 'INR', 'EUR'")


def convert_currency_func(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies."""
    # Simplified rates (in production, use a real API)
    rates = {
        ("USD", "INR"): 83.5, ("INR", "USD"): 0.012,
        ("EUR", "INR"): 91.2, ("INR", "EUR"): 0.011,
        ("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09,
        ("GBP", "INR"): 105.8, ("INR", "GBP"): 0.0095,
    }
    
    if from_currency == to_currency:
        return f"{amount} {from_currency} = {amount} {to_currency}"
    
    key = (from_currency.upper(), to_currency.upper())
    if key not in rates:
        return f"Conversion rate not available for {from_currency} → {to_currency}"
    
    converted = amount * rates[key]
    return f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency} (rate: {rates[key]})"


# Create tool with extra configuration
currency_tool = StructuredTool.from_function(
    func=convert_currency_func,
    name="currency_converter",
    description="Convert amounts between currencies (USD, INR, EUR, GBP). "
                "Use when the user asks about currency conversion or exchange rates.",
    args_schema=CurrencyInput,
    return_direct=False,  # If True, tool result goes directly to user (skips LLM)
)

# Test
print(currency_tool.invoke({
    "amount": 1000, 
    "from_currency": "USD", 
    "to_currency": "INR"
}))
# 1,000.00 USD = 83,500.00 INR (rate: 83.5)
```

### `return_direct=True` — Skip the LLM

```python
# Normal flow (return_direct=False):
# User → LLM → Tool → Result → LLM → Natural language answer → User

# With return_direct=True:
# User → LLM → Tool → Result → User directly (skips final LLM call)
# Useful for tools that already produce user-friendly output

direct_tool = StructuredTool.from_function(
    func=convert_currency_func,
    name="currency_converter",
    description="Convert between currencies.",
    args_schema=CurrencyInput,
    return_direct=True,  # Result goes straight to user
)
```

---

## Part 6: `BaseTool` Subclass — Full Control

For maximum flexibility — async support, custom validation, callbacks:

```python
from typing import Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class OrderLookupInput(BaseModel):
    """Input for order lookup."""
    order_id: str = Field(description="The order ID to look up, e.g., 'ORD-12345'")


class OrderLookupTool(BaseTool):
    """Look up customer orders by order ID."""
    
    name: str = "order_lookup"
    description: str = (
        "Look up a customer order by its order ID. "
        "Use when the user asks about an order's status, details, tracking, "
        "or delivery information. Input should be an order ID like 'ORD-12345'."
    )
    args_schema: Type[BaseModel] = OrderLookupInput
    
    # Custom attributes
    orders_db: dict = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simulated database
        self.orders_db = {
            "ORD-12345": {
                "customer": "Rahul Sharma",
                "items": ["Wireless Headphones", "USB-C Cable"],
                "total": 3298,
                "status": "shipped",
                "tracking": "IN8847562390",
                "estimated_delivery": "2025-09-15"
            },
            "ORD-12346": {
                "customer": "Priya Patel",
                "items": ["Laptop Stand", "Mechanical Keyboard"],
                "total": 5298,
                "status": "processing",
                "tracking": None,
                "estimated_delivery": "2025-09-18"
            },
            "ORD-12347": {
                "customer": "Amit Kumar",
                "items": ["Clean Code (Book)"],
                "total": 899,
                "status": "delivered",
                "tracking": "IN8847562401",
                "estimated_delivery": "2025-09-10"
            },
        }
    
    def _run(
        self, 
        order_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool — look up the order."""
        order_id = order_id.upper().strip()
        
        if order_id not in self.orders_db:
            return f"Order '{order_id}' not found. Please check the order ID and try again."
        
        order = self.orders_db[order_id]
        
        status_emoji = {
            "processing": "🔄", "shipped": "🚚", 
            "delivered": "✅", "cancelled": "❌"
        }
        
        result = f"Order {order_id}:\n"
        result += f"  👤 Customer: {order['customer']}\n"
        result += f"  📦 Items: {', '.join(order['items'])}\n"
        result += f"  💰 Total: ₹{order['total']:,}\n"
        result += f"  {status_emoji.get(order['status'], '❓')} Status: {order['status']}\n"
        
        if order['tracking']:
            result += f"  🔍 Tracking: {order['tracking']}\n"
        
        result += f"  📅 Est. Delivery: {order['estimated_delivery']}"
        
        return result


# Create and test
order_tool = OrderLookupTool()
print(order_tool.invoke({"order_id": "ORD-12345"}))
```

### When to Use Each Method

| Method | Use When | Complexity |
|--------|----------|------------|
| `@tool` decorator | Simple functions, quick prototyping | ⭐ Low |
| `StructuredTool.from_function` | Need custom name/description, `return_direct` | ⭐⭐ Medium |
| `BaseTool` subclass | Need async, callbacks, state, custom validation | ⭐⭐⭐ High |

**Rule of thumb**: Start with `@tool`. Graduate to `BaseTool` only when you need its features.

---

## Part 7: Error Handling in Tools

Tools MUST handle errors gracefully. A crashing tool = a crashing agent.

### The Pattern

```python
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol (e.g., 'AAPL', 'RELIANCE.NS').
    
    Use when the user asks about stock prices, market data, or share values.
    """
    try:
        # Validate input
        ticker = ticker.upper().strip()
        if not ticker:
            return "Error: Please provide a valid ticker symbol."
        if len(ticker) > 15:
            return "Error: Invalid ticker symbol (too long)."
        
        # Simulated stock data (in production, use yfinance or an API)
        stock_data = {
            "AAPL": {"price": 178.72, "change": +2.15, "change_pct": 1.22},
            "GOOGL": {"price": 141.80, "change": -0.54, "change_pct": -0.38},
            "RELIANCE.NS": {"price": 2487.50, "change": +35.20, "change_pct": 1.44},
            "TCS.NS": {"price": 3654.30, "change": -12.80, "change_pct": -0.35},
        }
        
        if ticker not in stock_data:
            return (
                f"Stock data not found for '{ticker}'. "
                f"Available tickers: {', '.join(stock_data.keys())}"
            )
        
        data = stock_data[ticker]
        direction = "📈" if data["change"] >= 0 else "📉"
        sign = "+" if data["change"] >= 0 else ""
        
        return (
            f"{direction} {ticker}: ₹{data['price']:,.2f}\n"
            f"   Change: {sign}{data['change']:.2f} ({sign}{data['change_pct']:.2f}%)"
        )
    
    except Exception as e:
        return f"Error fetching stock data for '{ticker}': {str(e)}"
```

### Error Handling Best Practices

```python
# ✅ DO: Return error messages as strings
@tool
def risky_tool(input: str) -> str:
    """..."""
    try:
        result = some_api_call(input)
        return str(result)
    except TimeoutError:
        return "Error: The service is not responding. Please try again later."
    except ValueError as e:
        return f"Error: Invalid input — {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred — {str(e)}"

# ❌ DON'T: Let exceptions bubble up
@tool
def bad_tool(input: str) -> str:
    """..."""
    result = some_api_call(input)  # If this fails, the entire agent crashes!
    return str(result)
```

---

## Part 8: Complete Project — Customer Support Toolkit

Let's build a comprehensive customer support system with multiple custom tools:

```python
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

load_dotenv()

# ============================================================
# TOOL 1: Order Lookup
# ============================================================

# Simulated database
ORDERS_DB = {
    "ORD-1001": {
        "customer": "Rahul Sharma", "email": "rahul@example.com",
        "items": [{"name": "Wireless Headphones", "qty": 1, "price": 2499}],
        "total": 2499, "status": "shipped",
        "tracking": "IN8847562390", "date": "2025-08-28",
    },
    "ORD-1002": {
        "customer": "Priya Patel", "email": "priya@example.com",
        "items": [
            {"name": "Laptop Stand", "qty": 1, "price": 1299},
            {"name": "USB-C Hub", "qty": 2, "price": 1899}
        ],
        "total": 5097, "status": "processing",
        "tracking": None, "date": "2025-08-30",
    },
    "ORD-1003": {
        "customer": "Amit Kumar", "email": "amit@example.com",
        "items": [{"name": "Mechanical Keyboard", "qty": 1, "price": 3999}],
        "total": 3999, "status": "delivered",
        "tracking": "IN8847562401", "date": "2025-08-25",
    },
}

TICKETS_DB = []
FAQS = {
    "return_policy": "Items can be returned within 30 days of delivery. Item must be unused and in original packaging. Refund is processed within 5-7 business days.",
    "shipping": "Standard shipping: 5-7 business days. Express shipping: 2-3 business days (₹199 extra). Free shipping on orders above ₹999.",
    "payment": "We accept Credit/Debit Cards, UPI, Net Banking, and COD (Cash on Delivery). EMI options available for orders above ₹3,000.",
    "warranty": "All electronics come with 1-year manufacturer warranty. Extended warranty available for purchase. Clothing has 6-month warranty against defects.",
    "cancellation": "Orders can be cancelled before shipping. Once shipped, you'll need to initiate a return. Cancellation refund processed within 2-3 business days.",
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up a customer order by its order ID.
    
    Use when the user asks about their order status, tracking, items,
    delivery, or any order-specific information.
    
    Args:
        order_id: The order ID (e.g., 'ORD-1001')
    """
    order_id = order_id.upper().strip()
    
    if order_id not in ORDERS_DB:
        # Try to find by partial match
        matches = [k for k in ORDERS_DB if order_id in k]
        if matches:
            return f"Order '{order_id}' not found. Did you mean: {', '.join(matches)}?"
        return f"Order '{order_id}' not found. Please verify your order ID."
    
    order = ORDERS_DB[order_id]
    status_map = {"processing": "🔄 Processing", "shipped": "🚚 Shipped", 
                  "delivered": "✅ Delivered", "cancelled": "❌ Cancelled"}
    
    items_str = "\n".join(
        f"    • {item['name']} x{item['qty']} — ₹{item['price']:,}" 
        for item in order["items"]
    )
    
    result = (
        f"📦 Order {order_id}\n"
        f"  Customer: {order['customer']}\n"
        f"  Date: {order['date']}\n"
        f"  Status: {status_map.get(order['status'], order['status'])}\n"
        f"  Items:\n{items_str}\n"
        f"  Total: ₹{order['total']:,}\n"
    )
    
    if order["tracking"]:
        result += f"  Tracking: {order['tracking']}\n"
    
    return result


# ============================================================
# TOOL 2: Search Orders by Customer
# ============================================================

@tool
def search_customer_orders(customer_name: str) -> str:
    """Search for all orders belonging to a customer by their name.
    
    Use when the user asks about their order history, all their orders,
    or when you need to find orders without a specific order ID.
    
    Args:
        customer_name: The customer's name (partial match supported)
    """
    name_lower = customer_name.lower()
    matching_orders = {
        oid: order for oid, order in ORDERS_DB.items()
        if name_lower in order["customer"].lower()
    }
    
    if not matching_orders:
        return f"No orders found for customer '{customer_name}'."
    
    result = f"Found {len(matching_orders)} order(s) for '{customer_name}':\n\n"
    for oid, order in matching_orders.items():
        result += f"  📦 {oid} | {order['date']} | ₹{order['total']:,} | {order['status']}\n"
        items = ", ".join(item["name"] for item in order["items"])
        result += f"     Items: {items}\n\n"
    
    return result


# ============================================================
# TOOL 3: Create Support Ticket
# ============================================================

class TicketInput(BaseModel):
    title: str = Field(description="Short title describing the issue")
    description: str = Field(description="Detailed description of the customer's problem")
    priority: str = Field(
        default="medium",
        description="Priority: 'low', 'medium', 'high', or 'critical'"
    )
    related_order: str = Field(
        default="",
        description="Related order ID if applicable (e.g., 'ORD-1001')"
    )


@tool(args_schema=TicketInput)
def create_support_ticket(
    title: str, 
    description: str, 
    priority: str = "medium",
    related_order: str = ""
) -> str:
    """Create a new support ticket for escalation.
    
    Use when:
    - The issue requires human agent intervention
    - The customer has a complex problem you can't resolve
    - The customer requests to speak with a human
    - There's a billing dispute or sensitive issue
    """
    ticket_id = f"TK-{len(TICKETS_DB) + 1:04d}"
    
    ticket = {
        "id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "related_order": related_order,
        "status": "open",
        "created_at": datetime.now().isoformat()
    }
    TICKETS_DB.append(ticket)
    
    priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
    
    return (
        f"✅ Support ticket created!\n"
        f"  Ticket ID: {ticket_id}\n"
        f"  Title: {title}\n"
        f"  Priority: {priority_emoji.get(priority, '⚪')} {priority}\n"
        f"  Status: Open\n"
        f"  {'Related Order: ' + related_order if related_order else ''}\n"
        f"\nA support agent will follow up within "
        f"{'1 hour' if priority in ('high', 'critical') else '24 hours'}."
    )


# ============================================================
# TOOL 4: FAQ Lookup
# ============================================================

@tool
def lookup_faq(topic: str) -> str:
    """Look up frequently asked questions about our store policies.
    
    Use when the user asks about:
    - Return/refund policy
    - Shipping information and timelines
    - Payment methods
    - Warranty details
    - Order cancellation
    
    Args:
        topic: The FAQ topic — 'return_policy', 'shipping', 'payment', 'warranty', 'cancellation'
    """
    topic_lower = topic.lower().strip().replace(" ", "_")
    
    # Try exact match first
    if topic_lower in FAQS:
        return f"📋 {topic_lower.replace('_', ' ').title()} Policy:\n\n{FAQS[topic_lower]}"
    
    # Try partial match
    matches = [k for k in FAQS if topic_lower in k or k in topic_lower]
    if matches:
        results = []
        for key in matches:
            results.append(f"📋 {key.replace('_', ' ').title()}:\n{FAQS[key]}")
        return "\n\n".join(results)
    
    available = ", ".join(k.replace("_", " ") for k in FAQS.keys())
    return f"FAQ topic '{topic}' not found. Available topics: {available}"


# ============================================================
# ASSISTANT CLASS
# ============================================================

class CustomerSupportBot:
    """AI-powered customer support bot with multiple tools."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        
        self.tools = [lookup_order, search_customer_orders, 
                      create_support_ticket, lookup_faq]
        self.tool_map = {t.name: t for t in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.messages = [
            SystemMessage(content="""You are a friendly, professional customer support agent for an online store.

Your capabilities:
1. **Order Lookup** — Look up order details, status, and tracking by order ID
2. **Customer Search** — Find all orders for a customer by name
3. **FAQ** — Answer questions about return policy, shipping, payment, warranty, cancellation
4. **Create Ticket** — Escalate complex issues by creating support tickets

Guidelines:
- Always be polite and empathetic
- Look up real data using tools — never make up order information
- If you can't resolve an issue, create a support ticket
- Provide clear, actionable next steps
- If the customer seems frustrated, acknowledge their frustration before solving
""")
        ]
    
    def chat(self, message: str) -> str:
        """Process a customer message and respond."""
        print(f"\n👤 Customer: {message}")
        
        self.messages.append(HumanMessage(content=message))
        
        for _ in range(5):  # Max 5 tool-calling rounds
            response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(response)
            
            if not response.tool_calls:
                print(f"🤖 Agent: {response.content}")
                return response.content
            
            for tc in response.tool_calls:
                print(f"   🔧 Using: {tc['name']}({tc['args']})")
                
                try:
                    result = self.tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {str(e)}"
                
                self.messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))
        
        return "I'm having trouble processing this. Let me create a ticket for you."
    
    def reset(self):
        """Reset conversation (keep system message)."""
        self.messages = self.messages[:1]


# --- Demo Conversation ---
bot = CustomerSupportBot()

# Scenario 1: Order inquiry
bot.chat("Hi, can you check the status of my order ORD-1001?")

# Scenario 2: Policy question
bot.chat("What's your return policy? And how long does shipping take?")

# Scenario 3: Customer history
bot.chat("Can you show me all orders for Priya Patel?")

# Scenario 4: Complex issue → escalation
bot.chat(
    "I received my keyboard from order ORD-1003 but some keys are not working. "
    "I need a replacement."
)

# Scenario 5: General chat
bot.chat("Thanks for your help! You've been great.")
```

---

## Common Mistakes

### Mistake 1: Missing or vague docstrings
```python
# ❌ The LLM can't decide when to use this
@tool
def fetch(x: str) -> str:
    return do_something(x)

# ✅ Clear, intent-focused docstring
@tool
def fetch_order_details(order_id: str) -> str:
    """Fetch order details by order ID (e.g., 'ORD-12345').
    Use when the user asks about order status, items, or delivery."""
    return do_something(order_id)
```

### Mistake 2: Not adding type hints
```python
# ❌ No type hints — LLM doesn't know what types to pass
@tool
def calculate(a, b):
    """Calculate sum."""
    return a + b

# ✅ Type hints generate proper JSON schema
@tool
def calculate(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b
```

### Mistake 3: Returning complex objects instead of strings
```python
# ❌ LLMs process text — a dict object confuses the response
@tool
def get_data(id: str) -> dict:
    """Get data."""
    return {"name": "Alice", "age": 30}

# ✅ Return a formatted string
@tool
def get_data(id: str) -> str:
    """Get data for a given ID."""
    data = {"name": "Alice", "age": 30}
    return f"Name: {data['name']}, Age: {data['age']}"
```

### Mistake 4: Not handling tool errors
```python
# ❌ Crashes the entire agent
@tool
def api_call(url: str) -> str:
    """Call an API."""
    return requests.get(url).json()  # What if 404? Timeout? Network error?

# ✅ Graceful error handling
@tool
def api_call(url: str) -> str:
    """Call an API and return the response."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Error calling API: {str(e)}"
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Write intent-focused docstrings | LLM decides which tool to call based on this |
| Add type hints to all parameters | Generates proper JSON schema for the LLM |
| Use Pydantic `Field(description=...)` for complex inputs | Gives parameter-level guidance to the LLM |
| Always return strings from tools | LLMs process text; convert dicts to formatted strings |
| Handle ALL errors inside tools | Return error message string instead of crashing |
| Include examples in descriptions | "e.g., 'ORD-12345'" helps the LLM format inputs |
| Test tools directly before binding to LLM | Verify `tool.invoke({...})` works correctly |
| Use `@tool` for 90% of cases | Only use `BaseTool` when you need async/callbacks |

---

## Interview Preparation

### Easy
**Q: How do you create a custom tool in LangChain?**

> Use the `@tool` decorator from `langchain_core.tools`. Decorate a function with `@tool` — the function name becomes the tool name, the docstring becomes the description (which the LLM reads to decide when to use it), and type hints are used to generate the input schema. The docstring is critical — it should clearly describe when and why to use the tool, not just what it does.

### Medium
**Q: What are the three ways to create tools in LangChain, and when would you use each?**

> (1) **`@tool` decorator** — simplest, covers 90% of cases. Use for straightforward functions where the function name and docstring provide enough context. (2) **`StructuredTool.from_function`** — use when you need to customize the tool name, description, or set `return_direct=True` separately from the function definition. Also useful when wrapping existing functions you can't modify. (3) **`BaseTool` subclass** — use for complex tools that need async execution, custom callbacks, internal state management, or sophisticated validation logic. Start simple with `@tool` and graduate to `BaseTool` only when needed.

### Hard
**Q: How would you design a tool system for a customer support AI that handles order lookups, refunds, and ticket creation?**

> Design principles: (1) **Separate read vs write tools** — `lookup_order` (read, low-risk) vs `process_refund` (write, high-risk). (2) **Use Pydantic schemas** for all tools with clear `Field(description=...)` for each parameter. (3) **Implement risk-based execution** — read tools auto-execute; write tools like `process_refund` require human approval via `interrupt_before` in LangGraph. (4) **Error handling** — every tool returns strings, catches all exceptions, and provides actionable error messages. (5) **Tool descriptions** include explicit "Use when..." and "Do NOT use when..." guidance. (6) **Audit logging** — log every tool call with timestamp, args, result, and user context. (7) **Fallback** — if the LLM can't resolve the issue with available tools, auto-escalate by creating a support ticket.

### Senior
**Q: What are the trade-offs between `return_direct=True` and the standard tool flow?**

> With `return_direct=True`, the tool's output goes directly to the user, skipping the LLM's interpretation step. **Pros**: Saves one LLM call (faster, cheaper), guarantees the tool's exact output reaches the user (useful for formatted data, tables, or precise calculations). **Cons**: The LLM can't synthesize tool results with other context, can't chain multiple tool calls (since the flow ends), and can't add helpful framing or explanation around raw results. Use `return_direct=True` for tools with already-polished output (e.g., formatted reports) and `False` (default) when the LLM needs to interpret, combine, or contextualize results.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **`@tool` decorator** | Simplest way to create a custom tool from a function |
| **Docstring** | The LLM reads this to decide WHEN to use the tool — make it count |
| **Type hints** | Generate the JSON schema the LLM uses for tool arguments |
| **Pydantic schema** | Maximum control over parameter descriptions, defaults, and constraints |
| **`StructuredTool`** | Middle-ground: more config than `@tool`, less code than `BaseTool` |
| **`BaseTool`** | Full control: async, callbacks, state, custom validation |
| **`return_direct`** | If True, tool result goes to user directly (skips LLM interpretation) |
| **Error handling** | Always catch exceptions inside tools; return error strings, never crash |

---

> [← Previous: Built-in Tools](chapter-31-builtin-tools.md) | [Next: Tool Calling Deep Dive →](chapter-33-tool-calling-deep-dive.md)
