# Agentic AI: A Hands-On Guide

You learn this best by running the code beside the text. The demo is a **Trip Budget Agent**. You give it a
goal such as *"Can I do a 3 day trip to Goa with a budget of 20000?"*, and it finds the prices, does the maths,
saves a note and answers.

```
agentic_ai_demo/
  agent.py     the loop (the heart of every agent)
  tools.py     the agent's "hands": price lookup, calculator, notes
  brains.py    the decision-maker: ScriptedBrain (offline) or ClaudeBrain (real LLM)
  test_agent.py
```

Run it: `python agentic_ai_demo/agent.py` | tests: `pytest agentic_ai_demo`

---

## 1. What is Agentic AI?

A **chatbot** answers one question once. An **agent** is given a **goal** and works toward it by repeatedly
deciding what to do, doing it, and looking at what happened.

> Agentic AI = a model that can **choose actions**, **use tools**, **observe results** and **loop**
> until a goal is met, with limited human steering in between.

| | Plain LLM call | Agent |
|---|---|---|
| Input | A prompt | A goal |
| Steps | 1 | As many as needed (bounded) |
| Can fetch fresh data or act | No | Yes, through tools |
| Who decides the next step | You | The model |
| Failure mode | Wrong text | Wrong *actions*, which is why guardrails matter |

It is a spectrum, not a switch:

1. **Prompt**: one call, no tools.
2. **Workflow**: a fixed pipeline of calls that *you* wrote (A then B then C).
3. **Agent**: the model picks the path at run time.
4. **Multi-agent**: several agents with different roles hand work to each other.

Rule of thumb: use the simplest level that works. Agents trade predictability for flexibility.

## 2. The four ingredients

| Ingredient | In the demo | Role |
|---|---|---|
| **Goal** | `history[0]` | What "done" means |
| **Brain** (the model) | `brains.py` | Chooses the next action from the current state |
| **Tools** | `tools.py` | The only way the agent touches the world |
| **Memory / state** | `history` list (+ `NOTES`) | What happened so far |

Wrapped in one **loop**, which is `run_agent()` in `agent.py`:

```
        ┌──────────────────────────────────────────┐
        ▼                                          │
   THINK (brain.decide(history))                   │
        │                                          │
   final answer? ──yes──► DONE                     │
        │ no                                       │
   ACT (run_tool)  ──►  OBSERVE (append result) ───┘
        │
   step limit hit? ──► STOP (guardrail)
```

This is sometimes called the **ReAct pattern** (Reason + Act). Almost every agent framework is this loop with
extra features.

## 3. Walkthrough: reading a real run

```
GOAL: Can I do a 3 day trip to Goa with a budget of 20000?

[1] THOUGHT : I don't know the hotel_per_night for goa yet, so I'll look it up.
[1] ACTION  : lookup_price({'city': 'goa', 'item': 'hotel_per_night'})
[1] OBSERVE : 3500
... (food 1200, transport 800)
[4] ACTION  : calculator({'expression': '3500*2 + 1200*3 + 800*3'})
[4] OBSERVE : 13000
[5] ACTION  : save_note({...})
[6] ANSWER  : Trip to Goa for 3 days costs about 13000 INR; within budget with 7000 to spare.
```

Things to notice:

- **Nobody scripted "5 tool calls".** The brain looked at the history each turn and decided what was still
  missing. Change the goal to Manali, 5 days, 10000 and the path adapts, ending with "over budget by 12500".
- **The model never does the arithmetic or knows the prices.** It delegates to tools. This is the main reason
  agents exist: LLMs are weak at exact maths and have no live data, so tools make up for both.
- **Every observation goes back into the history**, so the next decision can use it. State is how the agent
  "knows" anything between steps.

## 4. Code tour (in reading order)

### `tools.py`: tools are just functions plus a description
```python
SCHEMAS = [{"name": "calculator", "description": "...", "input_schema": {...}}]
```
The **schema** is what a real LLM sees. The model never sees your Python; it reads the name, the description
and the argument types, and then *asks* for a call. **Your code** executes it. Tool descriptions are
effectively prompts, so write them carefully.

Two safety habits live here:
- `calculator` parses the expression with `ast` instead of `eval()`, because model output is untrusted input.
- `run_tool` turns a bad tool name or bad arguments into an `ERROR:` **observation** instead of a crash. The
  agent can read the error and recover, which is a big part of why agents are resilient.

### `agent.py`: the loop
Read `run_agent()`. It is about 20 lines. Note the three decisions that make it an agent rather than a script:
1. The brain, not the code, picks the action.
2. Results are appended to `history` and fed back in.
3. The loop ends on a `final` decision **or** at `max_steps`.

### `brains.py`: the swappable brain
`ScriptedBrain` is plain `if` statements that imitate an LLM, so the demo runs free and offline and you can see
the mechanics. It exposes one method:

```python
decide(history) -> {"type": "tool_call", "name": ..., "args": ..., "thought": ...}
               or  {"type": "final", "text": ...}
```

`ClaudeBrain` implements the **same interface** with a real model using Anthropic's tool-use API: it sends the
history plus `SCHEMAS`, and the model replies either with text (done) or a `tool_use` block (call this tool).
Run it with:

```
pip install anthropic
export ANTHROPIC_API_KEY=...
python agentic_ai_demo/agent.py --brain claude "Can I do a 3 day trip to Jaipur with a budget of 9000?"
```

Because the interface is identical, you can swap the brain without touching the loop or tools. That separation
is the key architectural idea.

## 5. Guardrails and safety

The demo has three, and real systems need more:

| Guardrail | Where | Why |
|---|---|---|
| Step limit | `MAX_STEPS` | Stops runaway loops and runaway cost |
| Safe tool execution | `calculator`, `run_tool` | Never run model output blindly |
| Errors as observations | `run_tool` | Lets the agent self-correct |

Add these as agents get more powerful: least-privilege tools (read-only by default), **human approval** before
irreversible actions (payments, deletes, emails), logging of every step for audit, budget/time caps,
and defence against **prompt injection** (text returned by a tool, such as a web page, may try to give the
agent new instructions; treat tool output as data, not commands).

## 6. Memory

- **Short-term**: the `history` list. It lives for one run and is fed to the brain each step. It is limited by
  the model's context window, so long runs need summarising or trimming.
- **Long-term**: `NOTES` via `save_note`. It persists across steps (in a real app, a database or vector store)
  and would be read back at the start of future runs.

## 7. When agents are the right tool, and when they are not

Good fit: open-ended tasks where the steps aren't known in advance (research, debugging code, triaging
tickets, multi-system lookups).

Poor fit: fixed, repeatable processes. Use a plain workflow there; it is cheaper, faster and testable. Also be
careful when mistakes are costly and unrecoverable. Agents compound errors across steps, so every extra step
multiplies risk and cost.

## 8. Exercises (do these in order)

1. **Run it** on Goa, Jaipur and Manali with different budgets. Read the trace.
2. **Break it**: ask for "Paris". What does the observation say? Who handles it, and what does the final answer
   look like? (Hint: the scripted brain only knows three cities, so see what a real LLM would do instead.)
3. **Add a tool**: `lookup_weather(city)` returning fake data. Add it to `REGISTRY` and `SCHEMAS`, then teach
   `ScriptedBrain` to call it. (With `ClaudeBrain` you only edit `tools.py`; the model discovers the tool by itself.)
4. **Lower `max_steps` to 3** and watch the guardrail trigger.
5. **Add human approval**: before `save_note` runs, ask `input("Allow? [y/N]")`. This is the
   human-in-the-loop pattern.
6. **Go real**: run with `--brain claude` and compare its trace to the scripted one. Notice it may batch or
   reorder steps and phrase its thoughts differently.
7. **Stretch**: split into two agents, a *Researcher* that only fetches prices and a *Planner* that decides
   (multi-agent).

## 9. Glossary

- **Agent**: a model plus tools plus a loop working toward a goal.
- **Tool / function calling**: structured way for the model to request an action; your code executes it.
- **Observation**: the result of a tool call, fed back to the model.
- **ReAct**: reason, act, observe, repeat.
- **Context window**: how much history the model can see at once.
- **Orchestration**: the code around the model (loop, limits, retries).
- **Human-in-the-loop**: a person approves or steers at key steps.
- **Prompt injection**: malicious instructions hidden in content the agent reads.
- **Multi-agent**: several specialised agents collaborating.

## 10. Key takeaways

1. An agent is a **loop**: think, act, observe, repeat, with a stop condition.
2. The model decides; **tools do**; your **code enforces the rules**.
3. State (history) is what lets each step build on the last.
4. Start simple, add guardrails early, and add autonomy gradually.
