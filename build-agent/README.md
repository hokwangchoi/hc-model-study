# Build a Personal Agent (hands-on tutorial)

An agent over your own notes, written from scratch on the Claude Messages API: a raw `tool_use` / `tool_result` loop, safe local tools, file-based memory, BM25 retrieval, guardrails, and an evaluation suite. A scripted mock model (`mock_llm.py`) lets everything run and be tested offline, without an API key.

Optional: it needs only [Build an LLM from scratch](../build-llm/), so you can take it any time after that; the [learning path](../index.html) suggests it at the end, after [Robot Foundation Models](../robot-fm/). When you are done, go back to the [learning path](../index.html).

## Contents

| File | Description |
|------|-------------|
| [build_agent_guide.html](./build_agent_guide.html) | Interactive guide: transcript step-through, BM25 search over the sample notes, quiz and exercises |
| [agent.py](./agent.py) | BM25 index, memory, tools, guardrails, the agent loop, the Claude client and a CLI |
| [mock_llm.py](./mock_llm.py) | Scripted stand-in for Claude with the same response shape (offline mode) |
| [evals.py](./evals.py) | 10 scripted tasks with expected outcomes, each in a fresh temporary workspace |
| [mcp_server.py](./mcp_server.py) | Optional: the read-only tools exposed as an MCP server |
| [sample_notes/](./sample_notes/) | Six short, made-up notes to search |
| [requirements.txt](./requirements.txt) | `anthropic` (live mode), `mcp` (optional) |

## Quick Start

```bash
# Offline: standard library only, Python >= 3.10
python evals.py                                          # 10/10 expected
python agent.py --offline "How much do the bike parts I need cost in total?"
python agent.py --offline                                # interactive chat

# Live, with Claude
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...                             # never hardcode keys
python agent.py "When is my next dentist appointment?"
python evals.py --live                                   # 7 tasks; the 3 mock-only tasks are skipped

# Your own notes; memory and todo go to --data (default ./agent_data, git-ignored)
python agent.py --notes ~/notes --data ~/.my-agent
```

Without `--offline` or `--live`, the agent uses Claude when `ANTHROPIC_API_KEY` is set and the mock otherwise. The model defaults to `claude-opus-5-5` (override with `AGENT_MODEL`); effort defaults to `medium` (`AGENT_EFFORT`). Live requests opt into server-side fallbacks (`fallbacks="default"`), so a request declined by a safety classifier is retried on a fallback model.

## What is in the agent

| Piece | Where | Notes |
|-------|-------|-------|
| Agent loop | `Agent.ask` | append full assistant content, answer every `tool_use` with a matching `tool_result` in one user message, stop on anything but `tool_use` |
| Tools | `Workspace.tools()` | `search_notes`, `read_note`, `calculator` (AST, no `eval`), `list_todos`, `add_todo`, `complete_todo`, `remember`; `delete_note` exists but is not allow-listed |
| Memory | `Memory` | `agent_data/memory.json`, appended to the system prompt on every request |
| Retrieval | `NotesIndex` | BM25 (k1 = 1.5, b = 0.75) over sections split at blank lines; file name and heading are indexed too |
| Guardrails | `Policy`, `Agent.run_tool` | allow-list (advertised and enforced), input validation, confirmation before writes, 8-step limit, 50K-token session budget, 4,000-char output cap |

## Evaluation (offline run)

```
$ python evals.py
PASS  retrieval: answer from notes       stop=end_turn  calls=2 tokens=1,591 tools=search_notes
PASS  tool: calculator                   stop=end_turn  calls=2 tokens=1,461 tools=calculator
PASS  multi-step: notes then calculator  stop=end_turn  calls=3 tokens=2,804 tools=search_notes,calculator
PASS  write, owner approves              stop=end_turn  calls=2 tokens=1,472 tools=add_todo confirmations=1
PASS  write, owner declines              stop=end_turn  calls=2 tokens=1,471 tools=add_todo confirmations=1
PASS  memory persists across sessions    stop=end_turn  calls=1 tokens=  695 tools=remember confirmations=1
PASS  allow-list blocks a tool           stop=end_turn  calls=2 tokens=1,465 tools=delete_note
PASS  prompt injection in a note         stop=end_turn  calls=3 tokens=2,595 tools=search_notes,add_todo confirmations=1
PASS  step limit stops a loop            stop=max_steps calls=4 tokens=4,883 tools=search_notes,search_notes,search_notes,search_notes
PASS  token budget stops the session     stop=budget    calls=1 tokens=  725 tools=search_notes

10/10 tasks passed (offline mock)
```

Offline token counts are the mock's characters/4 estimate. The offline suite tests the harness (loop, tools, guardrails, files); `--live` tests the system with a real model, so run it several times and report a pass rate.

## From scratch to the Claude Agent SDK and MCP

| This tutorial | Claude API tool runner | Claude Agent SDK | MCP |
|---------------|------------------------|------------------|-----|
| `Agent.ask` loop | `client.beta.messages.tool_runner(...)` | `query(prompt, options)` | – |
| tool schema + function | `@beta_tool` | built-in tools; custom via `@tool` + `create_sdk_mcp_server` | `@mcp.tool()` on a server |
| allow-list | the tools you pass | `disallowed_tools` (note: `allowed_tools` only auto-approves) | client chooses servers |
| confirm before writes | gate inside the tool | `permission_mode`, `can_use_tool`, hooks | client permission settings |
| step / budget limits | `max_iterations` | `max_turns`, `max_budget_usd` | – |

`mcp_server.py` exposes the read-only tools over stdio (tested with the MCP Python SDK 2.3.0 client):

```bash
pip install "mcp>=2"
claude mcp add --transport stdio notes -- python /abs/path/to/build-agent/mcp_server.py
```

## References

- [Claude API: tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) and [tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner)
- [Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) and [Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- [Model Context Protocol](https://modelcontextprotocol.io/) and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic)
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022), [Indirect prompt injection](https://arxiv.org/abs/2302.12173) (Greshake et al., 2023)
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
