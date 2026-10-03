import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from agent import run_agent  # noqa: E402
from brains import ScriptedBrain  # noqa: E402
from tools import calculator, run_tool  # noqa: E402


def test_goa_within_budget():
    answer, history = run_agent("Can I do a 3 day trip to Goa with a budget of 20000?",
                                ScriptedBrain(), verbose=False)
    assert "within budget" in answer and "13000" in answer
    assert sum(h["kind"] == "action" for h in history) == 5  # 3 lookups + calc + note


def test_manali_over_budget():
    answer, _ = run_agent("5 day trip to Manali with a budget of 10000", ScriptedBrain(), verbose=False)
    assert "over budget" in answer


def test_step_limit_stops_the_loop():
    answer, _ = run_agent("Can I do a 3 day trip to Goa with a budget of 20000?",
                          ScriptedBrain(), max_steps=2, verbose=False)
    assert "Stopped after 2 steps" in answer


def test_calculator_rejects_code():
    assert calculator("2+3*4") == "14"
    assert calculator("__import__('os').system('ls')").startswith("ERROR")


def test_bad_tool_becomes_observation():
    assert run_tool("delete_everything", {}).startswith("ERROR")
    assert run_tool("calculator", {"wrong": 1}).startswith("ERROR")
