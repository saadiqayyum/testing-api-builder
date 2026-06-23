from typing import TypedDict
from langgraph.graph import StateGraph, END


# ── State ────────────────────────────────────────────────────────────────────

class DeduplicateState(TypedDict):
    input_word: str
    output_word: str


# ── Nodes ────────────────────────────────────────────────────────────────────

def deduplicate(state: DeduplicateState) -> DeduplicateState:
    """Remove consecutive duplicate letters from the input word."""
    word = state["input_word"]
    if not word:
        return {**state, "output_word": ""}

    result = [word[0]]
    for char in word[1:]:
        if char != result[-1]:
            result.append(char)

    return {**state, "output_word": "".join(result)}


# ── Graph ────────────────────────────────────────────────────────────────────

builder = StateGraph(DeduplicateState)

builder.add_node("deduplicate", deduplicate)

builder.set_entry_point("deduplicate")
builder.add_edge("deduplicate", END)

graph = builder.compile()
