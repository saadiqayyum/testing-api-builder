from typing import TypedDict
from langgraph.graph import StateGraph, END


class TemperatureState(TypedDict):
    celsius: float
    fahrenheit: float
    result_message: str


def convert_temperature(state: TemperatureState) -> TemperatureState:
    """Convert Celsius to Fahrenheit using the formula: F = (C × 9/5) + 32"""
    celsius = state["celsius"]
    fahrenheit = (celsius * 9 / 5) + 32
    message = f"{celsius}°C = {fahrenheit:.2f}°F"
    return {
        **state,
        "fahrenheit": fahrenheit,
        "result_message": message,
    }


# Build the graph
builder = StateGraph(TemperatureState)
builder.add_node("convert", convert_temperature)
builder.set_entry_point("convert")
builder.add_edge("convert", END)

graph = builder.compile()
