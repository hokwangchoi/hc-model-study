"""Split a lab file into notebook-like cells, and put the cells back together.

Shared by labs/serve.py (the local lab runner) and labs/tools/build_pages.py
(which generates the lab pages and notebooks). Standard library only.

A lab file is cut into consecutive segments; joining their `source` strings
gives the file back byte for byte:

    docstring   the module docstring (shown rendered at the top of the page)
    header      the imports (and anything else before the first def/class)
    banner      a "# ----- / # Step 1: ... / # -----" comment block
    code        one top-level def or class, with the comments directly above it
    main        the `if __name__ == "__main__":` block that runs the checker

Code and header cells also carry `code`: the source without trailing blank
lines and without the "# Hint N:" comments, which the pages and notebooks
show separately (one at a time). `assemble()` puts the hints back where they
were, so saving from the page never removes them from the learner's file.
"""

import ast
import re

HINT_RE = re.compile(r"^(?P<indent>[ \t]*)# Hint (?P<num>\d+)(?P<label>[^:\n]*):[ \t]?(?P<text>.*)$")
RULE_RE = re.compile(r"^#\s?-{10,}\s*$")
TODO_RE = re.compile(r"^\s*raise NotImplementedError\b")
LAB_FILE_RE = re.compile(r"^lab(\d{2})_[a-z0-9_]+\.py$")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
DEF_RE = re.compile(r"^(?:async\s+def|def|class)\s+(\w+)")
MAIN_RE = re.compile(r"""^if\s+__name__\s*==\s*['"]__main__['"]\s*:""")


def _leading_comment_start(lines, first):
    """Index of the first line of the comment block directly above line index `first`."""
    k = first
    while k > 0 and lines[k - 1].startswith("#"):
        k -= 1
    return k


def _banner_at(lines, k, end):
    """If lines[k:end] starts with a rule / title / rule banner, return (title, n_lines)."""
    if end - k >= 3 and RULE_RE.match(lines[k]) and RULE_RE.match(lines[k + 2]):
        title = lines[k + 1].strip()
        if title.startswith("#") and not RULE_RE.match(lines[k + 1]):
            return title.lstrip("#").strip(), 3
    return None


def _split_trailing(text):
    """(text without trailing blank lines but with one final newline, the trailing newlines)."""
    body = text.rstrip(" \t\n")
    if not body:
        return "", text
    rest = text[len(body):]
    if rest.startswith("\n"):
        rest = rest[1:]
    # trailing whitespace-only lines are kept as plain newlines
    return body + "\n", "\n" * rest.count("\n")


def _docstring_lines(lines):
    """Number of lines taken by a module docstring at the top of the file (0 if none)."""
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i == len(lines):
        return 0
    s = lines[i].lstrip()
    m = re.match(r"^[rRuU]?(\"\"\"|\'\'\')", s)
    if not m or lines[i][0].isspace():
        return 0
    quote = m.group(1)
    if quote in s[m.end():]:
        return i + 1
    for j in range(i + 1, len(lines)):
        if quote in lines[j]:
            return j + 1
    return 0


def split_segments(source):
    """Cut `source` into segments [{kind, id, title, start, end, source, node}] (0-based line ranges).

    Boundaries come from the text (top-level `def` / `class` / decorator /
    `if __name__ == "__main__":` lines at column 0), so a file with a syntax
    error inside a function still splits into the same cells. When the file
    parses, `node` is the matching ast node (for signatures and docstrings).
    """
    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source)
        nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    except SyntaxError:
        nodes = {}
    starts = []          # (line index, kind, id, title)
    pos = _docstring_lines(lines)
    if pos:
        starts.append((0, "docstring", "docstring", ""))
    header = None
    k = pos
    seen = set()
    while k < len(lines):
        ln = lines[k]
        kind = name = None
        first = k
        if ln.startswith("@"):
            j = k
            while j < len(lines) and not DEF_RE.match(lines[j]):
                j += 1
            m = DEF_RE.match(lines[j]) if j < len(lines) else None
            if m:
                kind, name = "code", m.group(1)
                k = j
        else:
            m = DEF_RE.match(ln)
            if m:
                kind, name = "code", m.group(1)
            elif MAIN_RE.match(ln):
                kind, name = "main", "main"
        if kind is None:
            if header is None and ln.strip() and not ln[0].isspace() and not ln.startswith("#") \
                    and not any(s[1] in ("code", "main") for s in starts):
                h = pos
                while h < len(lines) and not lines[h].strip():
                    h += 1
                header = (h, "header", "imports", "Imports")
                starts.append(header)
            k += 1
            continue
        top = _leading_comment_start(lines, first)
        if header is not None and top <= header[0]:
            top = first
        banner = _banner_at(lines, top, first)
        uid = name
        n = 2
        while uid in seen:            # the same name defined twice: keep ids unique
            uid = f"{name}~{n}"
            n += 1
        seen.add(uid)
        if banner:
            starts.append((top, "banner", "banner:" + uid, banner[0]))
            top += banner[1]
        starts.append((top, kind, uid, name if kind == "code" else ""))
        k += 1
    starts.sort(key=lambda s: s[0])
    segs = []
    for i, (start, kind, sid, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        if i == 0:
            start = 0                  # text before the first segment stays with it
        node = nodes.get(sid) if kind == "code" else None
        segs.append({"kind": kind, "id": sid, "title": title, "start": start, "end": end,
                     "source": "".join(lines[start:end]), "node": node})
    if not segs:
        segs.append({"kind": "header", "id": "imports", "title": "Imports", "start": 0,
                     "end": len(lines), "source": source, "node": None})
    return segs


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------
def _functions(tree):
    """Yield (qualname, FunctionDef) for top-level functions and methods of top-level classes."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{node.name}.{sub.name}", sub


def _docstring_end(fn):
    """1-based last line of the function's docstring, or None."""
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        return fn.body[0].end_lineno
    return None


def find_hint_blocks(lines):
    """Hint comment blocks in `lines`: [{start, end, items: [{num, label, text}]}] (0-based, end exclusive)."""
    blocks = []
    i = 0
    while i < len(lines):
        m = HINT_RE.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue
        indent = m.group("indent")
        start = i
        items = []
        while i < len(lines):
            m = HINT_RE.match(lines[i].rstrip("\n"))
            if not m or m.group("indent") != indent:
                break
            offset = len(lines[i].rstrip("\n")) - len(indent) - len(m.group("text"))
            text_lines = [m.group("text")]
            i += 1
            while i < len(lines):
                raw = lines[i].rstrip("\n")
                if not raw.startswith(indent + "#") or HINT_RE.match(raw):
                    break
                after = raw[len(indent):]           # "#   continuation"
                if not after.startswith("#  "):      # continuation lines are indented past "# "
                    break
                lead = len(after) - len(after.lstrip(" #")) if after.lstrip("#").strip() else len(after)
                cut = min(lead, offset)
                text_lines.append(after[cut:])
                i += 1
            items.append({"num": int(m.group("num")), "label": m.group("label").strip(),
                          "text": "\n".join(text_lines).rstrip()})
        blocks.append({"start": start, "end": i, "indent": indent, "items": items})
    return blocks


def _owner(tree, line1, lines=None):
    """Qualified name of the innermost function containing 1-based line `line1`, or ''."""
    if tree is None:
        return _owner_text(lines or [], line1 - 1)
    best = ""
    for qual, fn in _functions(tree):
        if fn.lineno <= line1 <= fn.end_lineno:
            best = qual
    return best


def _owner_text(lines, idx):
    """Like _owner, by indentation only (for code that does not parse)."""
    indent = len(lines[idx]) - len(lines[idx].lstrip()) if idx < len(lines) else 0
    name = def_indent = None
    for k in range(idx - 1, -1, -1):
        s = lines[k]
        if not s.strip():
            continue
        ind = len(s) - len(s.lstrip())
        m = re.match(r"^\s*(?:async\s+)?def\s+(\w+)", s)
        if name is None and m and ind < indent:
            name, def_indent = m.group(1), ind
            if ind == 0:
                return name
            continue
        if name is not None and ind < def_indent:
            m = re.match(r"^class\s+(\w+)", s)
            return f"{m.group(1)}.{name}" if m else name
    return name or ""


def extract_hints(code):
    """Remove the hint comments from `code`.

    Returns (code_without_hints, hints) where hints is a list of
    {func, anchor, indent, lines, items}: `lines` are the exact comment lines,
    `anchor` says where they sat ("docstring": right after the docstring, or
    "line:<text>": right after the line with that stripped text).
    """
    lines = code.splitlines(keepends=True)
    blocks = find_hint_blocks(lines)
    if not blocks:
        return code, []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    hints = []
    for b in blocks:
        func = _owner(tree, b["start"] + 1, lines)
        anchor = "line:"
        prev = b["start"] - 1
        while prev >= 0 and not lines[prev].strip():
            prev -= 1
        if tree is not None and func:
            fn = dict(_functions(tree))[func]
            doc_end = _docstring_end(fn)
            if doc_end is not None and prev + 1 == doc_end:
                anchor = "docstring"
        if anchor == "line:":
            anchor = "line:" + (lines[prev].strip() if prev >= 0 else "")
        hints.append({"func": func, "anchor": anchor, "indent": b["indent"],
                      "lines": [ln.rstrip("\n") for ln in lines[b["start"]:b["end"]]],
                      "items": b["items"]})
    keep = []
    drop = set()
    for b in blocks:
        drop.update(range(b["start"], b["end"]))
    for i, ln in enumerate(lines):
        if i not in drop:
            keep.append(ln)
    return "".join(keep), hints


def _insert_point_ast(fn, hint, lines):
    """0-based line index before which the hint block goes, using the parsed function."""
    first_body = fn.body[0].lineno if fn.body else fn.end_lineno + 1
    doc_end = _docstring_end(fn)
    if hint["anchor"].startswith("line:"):
        text = hint["anchor"][5:]
        lo = (doc_end if doc_end else first_body - 1)
        for k in range(lo, fn.end_lineno):
            if lines[k].strip() == text and text:
                return k + 1
    for k in range(first_body - 1, fn.end_lineno):
        if TODO_RE.match(lines[k]):
            return k
    if doc_end is not None:
        return doc_end
    return first_body - 1


def _insert_point_text(qual, lines):
    """Fallback when the code does not parse: find the def by text."""
    cls, _, name = qual.rpartition(".")
    lo = 0
    if cls:
        for k, ln in enumerate(lines):
            if re.match(rf"^class\s+{re.escape(cls)}\b", ln):
                lo = k
                break
        else:
            return None, None
    pat = re.compile(rf"^(\s*)(async\s+)?def\s+{re.escape(name)}\s*\(")
    for k in range(lo, len(lines)):
        m = pat.match(lines[k])
        if not m:
            continue
        def_indent = len(m.group(1))
        # end of the function: next non-blank line indented <= def_indent
        end = len(lines)
        for j in range(k + 1, len(lines)):
            s = lines[j]
            if s.strip() and (len(s) - len(s.lstrip())) <= def_indent and not s.lstrip().startswith("#"):
                end = j
                break
        for j in range(k + 1, end):
            if TODO_RE.match(lines[j]):
                return j, " " * (len(lines[j]) - len(lines[j].lstrip()))
        # after the signature (first line ending with ':') and a docstring if any
        j = k
        while j < end and not lines[j].split("#")[0].rstrip().endswith(":"):
            j += 1
        j += 1
        body_indent = " " * (def_indent + 4)
        n = j
        while n < end and not lines[n].strip():
            n += 1
        if n < end:
            body_indent = " " * (len(lines[n]) - len(lines[n].lstrip()))
            s = lines[n].lstrip()
            q = next((t for t in ('"""', "'''", 'r"""', "r'''") if s.startswith(t)), None)
            if q:
                quote = q.lstrip("r")
                rest = s[len(q):]
                if quote in rest:
                    return n + 1, body_indent
                for m2 in range(n + 1, end):
                    if quote in lines[m2]:
                        return m2 + 1, body_indent
        return j, body_indent
    return None, None


def reinsert_hints(code, hints):
    """Put hint blocks (from extract_hints) back into `code`.

    Blocks already present, and blocks whose function no longer exists, are
    skipped. Returns (code, dropped) where dropped lists the function names
    whose hints could not be placed.
    """
    if not hints:
        return code, []
    lines = code.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    present = {ln.strip() for ln in lines}
    try:
        tree = ast.parse(code)
        funcs = dict(_functions(tree))
    except SyntaxError:
        tree, funcs = None, {}
    inserts = []      # (index, [lines])
    dropped = []
    for h in hints:
        if h["lines"] and h["lines"][0].strip() in present:
            continue
        idx, indent = None, None
        if tree is not None:
            fn = funcs.get(h["func"])
            if fn is not None:
                idx = _insert_point_ast(fn, h, lines)
                ref = fn.body[0] if fn.body else fn
                indent = " " * (ref.col_offset if fn.body else fn.col_offset + 4)
        elif h["func"]:
            idx, indent = _insert_point_text(h["func"], lines)
        if idx is None:
            dropped.append(h["func"] or "?")
            continue
        block = [indent + ln.lstrip() + "\n" for ln in h["lines"]]
        inserts.append((idx, block))
    for idx, block in sorted(inserts, key=lambda t: t[0], reverse=True):
        lines[idx:idx] = block
    return "".join(lines), dropped


# ---------------------------------------------------------------------------
# Cells: what the runner sends to the page, and how a save is assembled
# ---------------------------------------------------------------------------
def _signature(node):
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}" + (f"({bases})" if bases else "")
    a = node.args
    parts = []
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        parts.append(arg.arg + ("" if d is None else "=" + ast.unparse(d)))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("" if d is None else "=" + ast.unparse(d)))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{node.name}({', '.join(parts)})"


def _todo_methods(node, lines):
    """For a class: [(qualname, signature, docstring)] of the methods that raise NotImplementedError."""
    out = []
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = lines[sub.lineno - 1:sub.end_lineno]
            if any(TODO_RE.match(s) for s in src):
                out.append((f"{node.name}.{sub.name}", _signature(sub), ast.get_docstring(sub) or ""))
    return out


def split_cells(source):
    """The cells of a lab file, as JSON-friendly dicts (see the module docstring)."""
    lines = source.splitlines(keepends=True)
    cells = []
    for seg in split_segments(source):
        cell = {"id": seg["id"], "kind": seg["kind"], "title": seg["title"], "source": seg["source"]}
        if seg["kind"] in ("code", "header"):
            body, sep = _split_trailing(seg["source"])
            code, hints = extract_hints(body)
            cell.update(code=code, sep=sep, hints=hints,
                        todo=any(TODO_RE.match(s) for s in body.splitlines()))
            node = seg["node"]
            if seg["kind"] == "code" and node is not None:
                cell["signature"] = _signature(node)
                cell["doc"] = ast.get_docstring(node) or ""
                cell["class"] = isinstance(node, ast.ClassDef)
                cell["methods"] = _todo_methods(node, lines) if isinstance(node, ast.ClassDef) else []
                cell["line"] = seg["start"] + 1
        cells.append(cell)
    return cells


def editable_cells(cells):
    return [c for c in cells if c["kind"] in ("code", "header")]


def all_hints(cells):
    out = []
    for c in cells:
        out.extend(c.get("hints", []))
    return out


def defined_functions(code):
    """Qualified names of the functions and methods `code` defines (by text if it does not parse)."""
    try:
        return {q for q, _ in _functions(ast.parse(code))}
    except SyntaxError:
        pass
    quals, cls = set(), None
    for ln in code.splitlines():
        m = re.match(r"^class\s+(\w+)", ln)
        if m:
            cls = m.group(1)
            continue
        m = re.match(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(", ln)
        if m:
            if not m.group(1):
                quals.add(m.group(2))
                cls = None
            elif cls:
                quals.add(f"{cls}.{m.group(2)}")
        elif ln.strip() and not ln[0].isspace() and not ln.startswith(("#", "@", ")")):
            cls = None
    return quals


def assemble(parts, hints):
    """Build a lab file from page parts.

    parts: [{"kind": "raw", "text": ...} | {"kind": "cell", "code": ..., "sep": "\\n\\n"}]
    hints: the hint blocks of the current file (all_hints(split_cells(current)))
    Returns (text, dropped_hint_functions).
    """
    out = []
    dropped = []
    remaining = list(hints)
    for p in parts:
        if p.get("kind") == "raw":
            out.append(str(p.get("text", "")))
            continue
        code = str(p.get("code", "")).replace("\r\n", "\n").replace("\r", "\n")
        code = code.rstrip(" \t\n")
        sep = p.get("sep", "\n\n")
        if not isinstance(sep, str) or set(sep) - {"\n"} or len(sep) > 4:
            sep = "\n\n"
        if not code:
            continue
        quals = defined_functions(code)
        mine = [h for h in remaining if h["func"] in quals]
        code, lost = reinsert_hints(code + "\n", mine)
        for h in mine:
            remaining.remove(h)
        dropped.extend(lost)
        out.append(code.rstrip("\n") + "\n" + sep)
    dropped.extend(h["func"] or "?" for h in remaining)
    text = "".join(out)
    text = text.rstrip("\n") + "\n"
    return text, dropped


def lab_number(filename):
    m = LAB_FILE_RE.match(filename)
    return int(m.group(1)) if m else None
