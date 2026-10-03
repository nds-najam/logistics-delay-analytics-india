"""Brains: the decision-maker plugged into the agent loop.

ScriptedBrain is a rule-based stand-in for an LLM so the demo runs offline.
It is NOT intelligent: it only reads the history and picks the next step, which
is exactly the interface a real LLM fills (see ClaudeBrain).
"""
import os
import re

from tools import SCHEMAS


class ScriptedBrain:
    ITEMS = ["hotel_per_night", "food_per_day", "transport_per_day"]

    def decide(self, history):
        goal = history[0]["text"]
        city = re.search(r"\b(goa|jaipur|manali)\b", goal, re.I)
        days = re.search(r"(\d+)\s*day", goal, re.I)
        budget = re.search(r"budget of\s*(\d+)", goal, re.I)
        if not (city and days and budget):
            return {"type": "final", "text": "I need a known city, number of days and a budget."}
        city, days, budget = city.group(1).lower(), int(days.group(1)), int(budget.group(1))

        # Observations so far, keyed by the action that produced them.
        actions = {h["id"]: h for h in history if h["kind"] == "action"}
        obs = [(actions[h["id"]], h["text"]) for h in history if h["kind"] == "observation"]
        prices = {a["args"].get("item"): float(t) for a, t in obs if a["name"] == "lookup_price"}
        total = next((float(t) for a, t in obs if a["name"] == "calculator" and not t.startswith("ERROR")), None)
        saved = any(a["name"] == "save_note" for a, _ in obs)

        for item in self.ITEMS:  # still missing facts -> go get them
            if item not in prices:
                return {"type": "tool_call", "name": "lookup_price", "args": {"city": city, "item": item},
                        "thought": f"I don't know the {item} for {city} yet, so I'll look it up."}
        if total is None:
            p = {k: f"{v:g}" for k, v in prices.items()}
            expr = (f"{p['hotel_per_night']}*{days - 1} + {p['food_per_day']}*{days}"
                    f" + {p['transport_per_day']}*{days}")
            return {"type": "tool_call", "name": "calculator", "args": {"expression": expr},
                    "thought": "I have all prices. I'll compute the total instead of guessing."}
        verdict = (f"Trip to {city.title()} for {days} days costs about {total:.0f} INR; "
                   + (f"within budget with {budget - total:.0f} to spare." if total <= budget
                      else f"over budget by {total - budget:.0f}."))
        if not saved:
            return {"type": "tool_call", "name": "save_note", "args": {"text": verdict},
                    "thought": "I'll remember the verdict for next time."}
        return {"type": "final", "text": verdict, "thought": "Goal achieved; nothing left to do."}


class ClaudeBrain:
    """Same interface, but a real model chooses the tools (Anthropic tool use)."""

    def __init__(self, model=None):
        import anthropic  # pip install anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")

    def decide(self, history):
        messages = []
        for h in history:
            if h["kind"] == "goal":
                messages.append({"role": "user", "content": h["text"]})
            elif h["kind"] == "action":
                messages.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": h["id"], "name": h["name"], "input": h["args"]}]})
            else:
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": h["id"], "content": h["text"]}]})
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, tools=SCHEMAS, messages=messages,
            system="You are a trip budget agent. Use tools for prices and arithmetic; never guess numbers.")
        text = " ".join(b.text for b in resp.content if b.type == "text")
        for b in resp.content:
            if b.type == "tool_use":
                return {"type": "tool_call", "name": b.name, "args": b.input, "thought": text}
        return {"type": "final", "text": text}
