"""The agent loop: decide -> act -> observe -> repeat, until done or out of steps.

    python agentic_ai_demo/agent.py "Can I do a 3 day trip to Goa with a budget of 20000?"
    python agentic_ai_demo/agent.py --brain claude "..."     # needs ANTHROPIC_API_KEY
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from brains import ClaudeBrain, ScriptedBrain  # noqa: E402
from tools import run_tool  # noqa: E402

MAX_STEPS = 10  # guardrail: an agent must never loop forever


def run_agent(goal: str, brain, max_steps: int = MAX_STEPS, verbose: bool = True):
    history = [{"kind": "goal", "text": goal}]  # the agent's working memory
    for step in range(1, max_steps + 1):
        decision = brain.decide(history)  # 1. THINK: brain picks the next move
        if verbose and decision.get("thought"):
            print(f"[{step}] THOUGHT : {decision['thought']}")
        if decision["type"] == "final":  # 4. DONE
            if verbose:
                print(f"[{step}] ANSWER  : {decision['text']}")
            return decision["text"], history
        call = {"kind": "action", "id": f"call_{step}", "name": decision["name"],
                "args": decision["args"], "thought": decision.get("thought", "")}
        history.append(call)
        if verbose:
            print(f"[{step}] ACTION  : {call['name']}({call['args']})")
        result = run_tool(call["name"], call["args"])  # 2. ACT: run the tool
        history.append({"kind": "observation", "id": call["id"], "text": result})  # 3. OBSERVE
        if verbose:
            print(f"[{step}] OBSERVE : {result}")
    msg = f"Stopped after {max_steps} steps without finishing."
    if verbose:
        print(msg)
    return msg, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("goal", nargs="?", default="Can I do a 3 day trip to Goa with a budget of 20000?")
    ap.add_argument("--brain", choices=["scripted", "claude"], default="scripted")
    args = ap.parse_args()
    print(f"GOAL: {args.goal}\n")
    run_agent(args.goal, ClaudeBrain() if args.brain == "claude" else ScriptedBrain())
