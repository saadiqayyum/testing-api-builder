"""
Mathematics Agent
-----------------
Input state  : { "a": float, "b": float, "operation": str }
Output state : { "result": float | str }
Supported ops: + - * /
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END


# ── State ──────────────────────────────────────────────────────────────────────

class MathState(TypedDict):
    a: float
    b: float
    operation: str
    result: float | str   # str is used for error messages


# ── Node ───────────────────────────────────────────────────────────────────────

def calculate(state: MathState) -> MathState:
    a   = state["a"]
    b   = state["b"]
    op  = state["operation"].strip()

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            result = "Error: Division by zero is not allowed."
        else:
            result = a / b
    else:
        result = f"Error: Unsupported operation '{op}'. Use one of: + - * /"

    return {**state, "result": result}


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(MathState)

    builder.add_node("calculate", calculate)

    builder.set_entry_point("calculate")
    builder.add_edge("calculate", END)

    return builder.compile()


graph = build_graph()
