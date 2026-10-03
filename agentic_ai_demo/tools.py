"""Tools: the agent's hands. Each tool is a plain Python function plus a schema
that tells the model what it does and which arguments it takes."""
import ast
import operator

PRICES = {  # fake price list (INR), stands in for a real API or database
    "goa": {"hotel_per_night": 3500, "food_per_day": 1200, "transport_per_day": 800},
    "jaipur": {"hotel_per_night": 2800, "food_per_day": 900, "transport_per_day": 600},
    "manali": {"hotel_per_night": 3000, "food_per_day": 1000, "transport_per_day": 1100},
}

NOTES = []  # long-lived "memory" the agent can write to


def lookup_price(city: str, item: str) -> str:
    city_prices = PRICES.get(city.lower())
    if city_prices is None:
        return f"ERROR: unknown city '{city}'. Known: {', '.join(PRICES)}"
    if item not in city_prices:
        return f"ERROR: unknown item '{item}'. Known: {', '.join(city_prices)}"
    return str(city_prices[item])


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only + - * / and numbers are allowed")


def calculator(expression: str) -> str:
    # Never eval() model output directly: it is untrusted input.
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"ERROR: {exc}"


def save_note(text: str) -> str:
    NOTES.append(text)
    return f"saved note #{len(NOTES)}"


REGISTRY = {"lookup_price": lookup_price, "calculator": calculator, "save_note": save_note}

SCHEMAS = [
    {"name": "lookup_price",
     "description": "Get a price in INR. item is hotel_per_night, food_per_day or transport_per_day.",
     "input_schema": {"type": "object",
                      "properties": {"city": {"type": "string"}, "item": {"type": "string"}},
                      "required": ["city", "item"]}},
    {"name": "calculator",
     "description": "Evaluate an arithmetic expression such as '3500*2 + 1200*3'.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "save_note",
     "description": "Save a short note (e.g. the final verdict) to memory.",
     "input_schema": {"type": "object",
                      "properties": {"text": {"type": "string"}},
                      "required": ["text"]}},
]


def run_tool(name: str, args: dict) -> str:
    """Safe dispatcher: bad tool names or arguments become observations, not crashes."""
    fn = REGISTRY.get(name)
    if fn is None:
        return f"ERROR: no such tool '{name}'. Available: {', '.join(REGISTRY)}"
    try:
        return fn(**args)
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
