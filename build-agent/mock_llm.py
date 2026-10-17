"""
A scripted stand-in for Claude so the agent runs and can be tested offline, without an API key.

It speaks the same wire format as the Messages API (content blocks of type "text" / "tool_use",
a stop_reason, and usage token counts) and follows a few hand-written rules. It is NOT a
language model: it exists to exercise the agent loop, the tools, the guardrails and the evals.
Two rules deliberately misbehave, so the guardrails have something to catch:
  - asked to delete something, it calls a tool that is not on the allow-list;
  - when a note contains an instruction to add a todo, it falls for it (prompt injection).
"""

import json
import re
from types import SimpleNamespace as NS

PRICE = re.compile(r"\b\d+\.\d{2}\b")
INJECTED_TODO = re.compile(r"add '([^']+)' to (?:the|my) todo", re.I)


def get(block, key, default=None):
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"the", "a", "my", "is", "of", "to", "do", "i", "what", "when"}


class MockLLM:
    def __init__(self):
        self.n = 0  # for unique tool_use ids

    # ---- wire format helpers ----
    def _response(self, blocks, stop_reason, system, tools, messages):
        prompt_chars = len(system) + len(json.dumps(tools)) + len(json.dumps(messages, default=str))
        out_chars = len(json.dumps(blocks, default=lambda o: o.__dict__))
        usage = NS(input_tokens=prompt_chars // 4, output_tokens=out_chars // 4)  # ~4 chars per token
        return NS(content=blocks, stop_reason=stop_reason, usage=usage)

    def _tool(self, name, args, say=None):
        self.n += 1
        blocks = [NS(type="text", text=say)] if say else []
        return blocks + [NS(type="tool_use", id=f"toolu_mock_{self.n:03d}", name=name, input=args)], "tool_use"

    @staticmethod
    def _answer(text):
        return [NS(type="text", text=text)], "end_turn"

    # ---- the "policy" ----
    def create(self, *, system, tools, messages, max_tokens=None):
        last = messages[-1]
        if isinstance(last["content"], list) and get(last["content"][0], "type") == "tool_result":
            blocks, stop = self._after_tool(system, messages)
        else:
            blocks, stop = self._route(last["content"], system)
        return self._response(blocks, stop, system, tools, messages)

    def _route(self, q, system):
        ql = q.lower().strip()
        if m := re.search(r"\bremember (?:that )?(.+)", q, re.I):
            return self._tool("remember", {"fact": m.group(1).strip().rstrip(".")})
        if m := re.search(r"(?:add|put) ['\"]?(.+?)['\"]? (?:to|on) my (?:todo|to-do)", q, re.I):
            return self._tool("add_todo", {"text": m.group(1)})
        if m := re.search(r"\bremind me to (.+)", q, re.I):
            return self._tool("add_todo", {"text": m.group(1).rstrip(".")})
        if re.search(r"(?:todo|to-do) list\??$|\bmy todos\b", ql):
            return self._tool("list_todos", {})
        if m := re.search(r"(?:mark|complete|finish)\D*(\d+)", ql):
            return self._tool("complete_todo", {"number": int(m.group(1))})
        if m := re.search(r"\b(?:delete|remove) (?:my )?(\w+) note", ql):
            return self._tool("delete_note", {"path": f"{m.group(1)}.md"})  # not allow-listed: will be blocked
        if m := re.search(r"([\d.]+)\s*% of ([\d.]+)", ql):
            return self._tool("calculator", {"expression": f"{m.group(1)} / 100 * {m.group(2)}"})
        if m := re.search(r"[\d.]+(?:\s*[-+*/^]\s*\(?[\d.]+\)?)+", ql):
            return self._tool("calculator", {"expression": m.group(0)})
        if re.search(r"\bdo i (?:like|prefer|use)\b|\bmy favou?rite\b|\bdo you remember\b", ql):
            memory = system.split("Long-term memory", 1)[-1].splitlines()[1:]
            best = max(memory, key=lambda f: len(words(f) & words(q)), default="")
            if best.startswith("- ") and words(best) & words(q):
                return self._answer(f"From memory: {best[2:]}.")
            return self._answer("I don't have that in memory.")
        return self._tool("search_notes", {"query": q}, say="Let me check your notes.")

    def _after_tool(self, system, messages):
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user" and isinstance(m["content"], str))
        call = next(b for b in messages[-2]["content"] if get(b, "type") == "tool_use")
        result = messages[-1]["content"][0]
        name, args, out, failed = get(call, "name"), get(call, "input"), get(result, "content"), get(result, "is_error")
        earlier = [get(b, "name") for m in messages if m["role"] == "assistant" for b in m["content"]
                   if get(b, "type") == "tool_use"]

        if name == "search_notes":
            if "keep searching" in question.lower():           # a model stuck in a loop
                return self._tool("search_notes", {"query": args["query"]})
            if out == "no matching notes":
                return self._answer("I couldn't find anything about that in your notes.")
            hits = json.loads(out)
            for h in hits:                                    # gullible: obeys text found inside a note
                m = INJECTED_TODO.search(h["text"])
                if m and "add_todo" not in earlier:
                    return self._tool("add_todo", {"text": m.group(1)})
            top = hits[0]
            if re.search(r"\b(total|sum|altogether|in all)\b", question.lower()):
                prices = PRICE.findall(top["text"])
                if prices:
                    return self._tool("calculator", {"expression": " + ".join(prices)})
            return self._answer(self._best_sentence(top, question))

        if name == "calculator":
            if failed:
                return self._answer(f"I couldn't compute that: {out}")
            src = self._last_search(messages)
            where = f" (prices from {src['file']})" if src else ""
            return self._answer(f"{args['expression']} = {out}{where}.")
        if name == "add_todo":
            if failed:
                src = self._last_search(messages)  # declined injected todo: still answer the real question
                tail = f" {self._best_sentence(src, question)}" if src else ""
                return self._answer(f"OK, I did not add \"{args['text']}\".{tail}")
            return self._answer(f"Added \"{args['text']}\" to your todo list ({out}).")
        if name == "remember":
            return self._answer("OK, I won't save that." if failed else f"Got it. I'll remember that {args['fact']}.")
        if name == "list_todos":
            return self._answer(f"Your todo list:\n{out}")
        if failed:
            return self._answer(f"I can't do that: {out}.")
        return self._answer(f"Done: {out}.")

    @staticmethod
    def _last_search(messages):
        for m in reversed(messages):
            if m["role"] == "user" and isinstance(m["content"], list):
                c = get(m["content"][0], "content")
                if isinstance(c, str) and c.startswith("["):
                    return json.loads(c)[0]
        return None

    @staticmethod
    def _best_sentence(hit, question):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", hit["text"]) if s.strip()]
        safe = [s for s in sentences if not INJECTED_TODO.search(s)] or sentences
        best = max(safe, key=lambda s: len(words(s) & words(question + " " + hit["section"])))
        return f"From {hit['file']} ({hit['section']}): {best}"
