#!/usr/bin/env python3
"""Generate the lab pages and notebooks from the lab files, so they stay in sync.

    python labs/tools/build_pages.py           # write the files below
    python labs/tools/build_pages.py --check   # exit 1 if any of them is out of date

Writes
    labs/index.html                     the lab hub (rendered from labs/README.md)
    labs/lab01.html ... lab10.html      one page per lab (maths, hints, code cells)
    labs/notebooks/labNN_<name>.ipynb   one Jupyter / Colab notebook per lab

Reads labs/labNN_*.py (run it on the untouched skeletons: the code cells are
embedded as they are), labs/README.md (order, time, guide mapping) and
labs/check.py (lab titles). Standard library only; the cell splitting is
shared with labs/serve.py through labs/tools/labcells.py.
"""

import argparse
import ast
import html
import inspect
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
LABS = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import labcells  # noqa: E402

REPO = "hokwangchoi/hc-model-study"
BRANCH = "master"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/labs/notebooks/{{}}"
RAW_CHECK = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/labs/check.py"
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:'
         'ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=DM+Sans:wght@400;500;600;700&display=swap" '
         'rel="stylesheet">')

# A starting point for the free scratch cell of each lab (not saved to the lab file).
SCRATCH = {
    1: "# Try your functions here (Run the cells above first).\n"
       "print(causal_mask(2, 4))\n"
       "x = torch.randn(2, 5)\n"
       "print(softmax(x).sum(-1))           # rows sum to 1\n"
       "q = k = v = torch.randn(1, 2, 3, 4)  # [B, h, N, d_k]\n"
       "out, w = scaled_dot_product_attention(q, k, v, causal_mask(3))\n"
       "out.shape, w.shape",
    2: "# Try your modules here (Run the cells above first).\n"
       "x = torch.randn(2, 5, 16)\n"
       "print(LayerNorm(16)(x).mean(-1)[0])   # ~0 per token\n"
       "blk = Block(16, 4, 64)\n"
       "print(blk(x).shape)\n"
       "count_block_params(768, 12, 3072)     # GPT-2 small block: 7,087,872",
    3: "# Try the model here (Run the cells above first).\n"
       "torch.manual_seed(0)\n"
       "model = GPT(GPTConfig()).eval()\n"
       "idx = torch.randint(0, 64, (1, 5))\n"
       "logits, presents = model(idx)\n"
       "print(logits.shape, presents[0][0].shape)\n"
       "generate(model, idx, 8)",
    4: "# Try the pieces here (Run the cells above first).\n"
       "img = torch.randn(2, 3, 32, 32)\n"
       "print(patchify(img, 8).shape)         # [2, 16, 192]\n"
       "trace = {}\n"
       "logits = ViT(ViTConfig())(img, trace)\n"
       "trace",
    5: "# Try the pieces here (Run the cells above first).\n"
       "print(count_image_tokens(336, 336, 14))           # LLaVA-1.5: 576\n"
       "print(count_image_tokens(448, 448, 14, merge=2))  # 256\n"
       "print(joint_attention_mask(2, 3, image_bidirectional=True).int())\n"
       "x = torch.randn(1, 16, 8)\n"
       "pixel_shuffle_merge(x, 4, 4).shape",
    6: "# Try the pieces here (Run the cells above first).\n"
       "betas = linear_beta_schedule()\n"
       "abar = alpha_bars_from_betas(betas)\n"
       "print(abar[[0, 250, 500, 999]])\n"
       "x0 = make_ring_data(4)\n"
       "q_sample(x0, torch.tensor([0, 250, 500, 999]), torch.randn(4, 2), abar)",
    7: "# Try the pieces here (Run the cells above first).\n"
       "b, L, D, N = 1, 6, 2, 3\n"
       "x, delta = torch.randn(b, L, D), torch.rand(b, L, D)\n"
       "A = -torch.rand(D, N) - 0.5\n"
       "B, C = torch.randn(b, L, N), torch.randn(b, L, N)\n"
       "y_seq, h = selective_scan_sequential(x, delta, A, B, C)\n"
       "y_par = selective_scan_parallel(x, delta, A, B, C)\n"
       "(y_seq - y_par).abs().max()",
    8: "# Try the routers here (Run the cells above first).\n"
       "logits = torch.randn(6, 4)              # 6 tokens, 4 experts\n"
       "w, idx, probs = softmax_topk_router(logits, k=2)\n"
       "print(idx)\n"
       "print(load_balancing_loss(probs, idx))\n"
       "expert_capacity(6, 4, 2, 1.25)",
    9: "# Try the geometry here (Run the cells above first).\n"
       "K = make_intrinsics(500.0, 800, 448)\n"
       "T = make_ego_to_cam(0.0, 1.6)\n"
       "uv, depth = project_points(torch.tensor([[10.0, 2.0, 0.0]]), K, T)\n"
       "uv, depth                                # guide: (300, 304), depth 10",
    10: "# Try the pieces here (Run the cells above first).\n"
        "a = mode_chunks()[:1]                    # [1, H, 2]\n"
        "eps = torch.randn_like(a)\n"
        "print(interpolate(eps, a, torch.tensor([0.5])).shape)\n"
        "print(chunk_trajectory(torch.arange(10.0)[:, None], horizon=4).shape)\n"
        "rollout_chunks(lambda step: torch.full((4, 1), float(step)), n_actions=6, n_exec=3)",
}


def esc(s):
    return html.escape(s, quote=True)


# ===========================================================================
# Inline formatting of docstring prose
# ===========================================================================
INLINE_RE = re.compile(r"""
    (?P<tick>`[^`\n]+`)
  | (?P<index>\b[A-Za-z_]\w*\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\](?:\[[^\[\]\n]*\])*)
  | (?P<shape>\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\])
  | (?P<call>\b[A-Za-z_][\w.]*\((?:[^()\n]|\((?:[^()\n]|\([^()\n]*\))*\))*\)(?:_\w+)?)
  | (?P<dotted>\b(?:torch|F|nn|np|math|self)(?:\.[A-Za-z_]\w*)+)
  | (?P<ident>\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b)
  | (?P<kw>\b(?:True|False|None)\b)
""", re.X)


def prose(text):
    """Escape prose and put identifiers, calls and shapes in <code>."""
    out, pos = [], 0
    for m in INLINE_RE.finditer(text):
        out.append(esc(text[pos:m.start()]).replace("-&gt;", "→"))
        tok = m.group(0)
        if m.group("tick"):
            tok = tok[1:-1]
        if m.group("shape") and not re.search(r"[A-Za-z0-9]", tok):
            out.append(esc(tok))
        else:
            out.append(f"<code>{esc(tok)}</code>")
        pos = m.end()
    out.append(esc(text[pos:]).replace("-&gt;", "→"))
    return "".join(out)


SHAPE_IN_PRE = re.compile(r"(?<![\w\])])\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\]")


def mono(text):
    """Escape a preformatted block; colour the tensor shapes."""
    out, pos = [], 0
    for m in SHAPE_IN_PRE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        tok = m.group(0)
        if re.fullmatch(r"\[[A-Za-z0-9_ ,*+/.()\-]+\]", tok) and re.search(r"[A-Za-z]", tok):
            out.append(f'<span class="sh">{esc(tok)}</span>')
        else:
            out.append(esc(tok))
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out).replace("-&gt;", "→")


def indent_of(line):
    return len(line) - len(line.lstrip(" "))


def dedent(lines):
    real = [ln for ln in lines if ln.strip()]
    n = min((indent_of(ln) for ln in real), default=0)
    return [ln[n:] if ln.strip() else "" for ln in lines]


def strip_blank(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


# ===========================================================================
# Function docstrings -> HTML
# ===========================================================================
ITEM_RE = re.compile(r"^(\d+[a-z]?)\.\s+(.*)$")


def render_doc(doc):
    """Paragraphs, numbered lists, and indented blocks (as code) of a function docstring."""
    lines = doc.split("\n")
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if indent_of(ln) > 0:
            block = []
            while i < len(lines) and (not lines[i].strip() or indent_of(lines[i]) > 0):
                block.append(lines[i])
                i += 1
            out.append(f'<pre class="doc-pre">{mono(chr(10).join(strip_blank(dedent(block))))}</pre>')
            continue
        if ITEM_RE.match(ln):
            items = []
            while i < len(lines):
                m = ITEM_RE.match(lines[i])
                if m:
                    items.append(m.group(2).strip())
                elif lines[i].strip() and 0 < indent_of(lines[i]) <= 4 and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            out.append("<ol>" + "".join(f"<li>{prose(t)}</li>" for t in items) + "</ol>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and indent_of(lines[i]) == 0 and not ITEM_RE.match(lines[i]):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        nxt = lines[i] if i < len(lines) else ""
        cls = ' class="doc-label"' if len(para) == 1 and text.endswith(":") and len(text) < 40 \
            and nxt.strip() and indent_of(nxt) > 0 else ""
        out.append(f"<p{cls}>{prose(text)}</p>")
    return "\n".join(out)


# ===========================================================================
# The module docstring: goal, maths, reading list
# ===========================================================================
def parse_module_doc(doc):
    lines = doc.split("\n")
    title = lines[0].strip()
    sections = []
    cur = None
    i = 1
    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if ln.strip() and re.fullmatch(r"-{3,}", nxt.strip()) and indent_of(ln) == 0:
            cur = {"head": ln.strip(), "lines": []}
            sections.append(cur)
            i += 2
            continue
        if ln.startswith("Run:"):
            cur = {"head": "Run", "lines": []}
            sections.append(cur)
        if cur is not None:
            cur["lines"].append(ln)
        i += 1
    for s in sections:
        s["lines"] = strip_blank(s["lines"])
    return title, sections


def section(sections, prefix):
    for s in sections:
        if s["head"].lower().startswith(prefix.lower()):
            return s
    return None


def render_goal(lines):
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        if lines[i].lstrip().startswith("* "):
            items = []
            while i < len(lines) and lines[i].strip():
                if lines[i].lstrip().startswith("* "):
                    items.append(lines[i].lstrip()[2:].strip())
                elif items:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append('<ul class="lead-list">' + "".join(f"<li>{prose(t)}</li>" for t in items) + "</ul>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith("* "):
            para.append(lines[i].strip())
            i += 1
        out.append(f'<p class="subtitle">{prose(" ".join(para))}</p>')
    return "\n".join(out)


def _prose_like(lines):
    """True if a run of lines reads as sentences rather than formulas."""
    text = " ".join(ln.strip() for ln in lines)
    if any(re.search(r"\S {3,}\S", ln.strip()) for ln in lines):
        return False                 # column-aligned: a formula table
    words = re.findall(r"[A-Za-z]+", text)
    toks = text.split()
    if not toks:
        return False
    plain = [w for w in words if len(w) >= 3 and w.lower() == w]
    return len(plain) >= 4 and len(plain) / len(toks) >= 0.45


def render_step_body(lines):
    """Continuation lines of a numbered maths step: prose runs as text, the rest as formulas."""
    lines = strip_blank(dedent(lines))
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        run = []
        base = indent_of(lines[i])
        while i < len(lines) and lines[i].strip() and indent_of(lines[i]) == base:
            run.append(lines[i])
            i += 1
        formula_start = re.match(r"^[\w\[\], '^*]{1,24}\s=\s", run[0].strip())
        if _prose_like(run) and base == 0 and not formula_start:
            out.append(f"<p>{prose(' '.join(ln.strip() for ln in run))}</p>")
        else:
            block = run
            while i < len(lines) and (not lines[i].strip() or indent_of(lines[i]) > base):
                block.append(lines[i])
                i += 1
            out.append(f'<pre class="math-pre">{mono(chr(10).join(strip_blank(dedent(block))))}</pre>')
    return "\n".join(out)


def render_maths(lines):
    """The 'maths, step by step' section: paragraphs, formula blocks and a numbered step list."""
    lines = dedent(lines)
    out = []
    steps = []

    def flush_steps():
        if steps:
            out.append('<ol class="msteps">' + "".join(steps) + "</ol>")
            steps.clear()

    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        ind = indent_of(ln)
        m = ITEM_RE.match(ln.strip())
        if m and ind > 0:
            body = []
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > ind):
                body.append(lines[j])
                j += 1
            head = m.group(2)
            parts = re.split(r"\s{2,}", head, maxsplit=1)
            colon = re.match(r"^([^:=]{3,48}):\s+(\S.*=.*)$", head)
            if len(parts) == 2 and len(parts[0]) <= 48:
                title, formula = parts
            elif colon:                                  # "Projector (LLaVA-1.5): z = W2 GELU(...)"
                title, formula = colon.group(1) + ":", colon.group(2)
            elif head.endswith(":") or len(head) <= 64:
                title, formula = head, ""
            elif _prose_like([head]):
                title, formula = "", ""
            else:
                title, formula = "", head
            html_ = f'<li><span class="sn">{esc(m.group(1))}</span>'
            if title:
                html_ += f'<div class="st">{prose(title)}</div>'
            inner = ""
            if formula:
                inner += f'<pre class="math-pre">{mono(formula.strip())}</pre>'
            elif not title:                    # a sentence that may go on in the next lines
                body = [head] + dedent(strip_blank(body))
            inner += render_step_body(body)
            html_ += f"<div>{inner}</div></li>"
            steps.append(html_)
            i = j
            continue
        flush_steps()
        if ind == 0:
            para = []
            while i < len(lines) and lines[i].strip() and indent_of(lines[i]) == 0:
                para.append(lines[i].strip())
                i += 1
            text = " ".join(para)
            nxt_indented = i < len(lines) and lines[i].strip() and indent_of(lines[i]) > 0
            if len(para) == 1 and (text.endswith(":") or (nxt_indented and len(text) < 60)):
                out.append(f'<p class="label">{prose(text.rstrip(":"))}</p>')
            else:
                out.append(f'<p class="para">{prose(text)}</p>')
            continue
        block = []
        while i < len(lines) and (not lines[i].strip() or indent_of(lines[i]) > 0):
            if lines[i].strip() and ITEM_RE.match(lines[i].strip()):
                break
            block.append(lines[i])
            i += 1
        out.append(f'<pre class="math-pre">{mono(chr(10).join(strip_blank(dedent(block))))}</pre>')
    flush_steps()
    return "\n".join(out)


def parse_readlist(lines):
    """Read-alongside entries: [("label", text)] or [("link", href, desc)]."""
    entries = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = re.match(r"^(\.\./\S+)\s*(.*)$", s)
        if m:
            entries.append(["link", m.group(1), m.group(2).strip()])
        elif entries and entries[-1][0] == "link" and indent_of(ln) > 4:
            entries[-1][2] = (entries[-1][2] + " " + s).strip()
        else:
            entries.append(["label", s.rstrip(":")])
    return entries


def link_label(href):
    path, _, frag = href.partition("#")
    name = Path(path).name
    nice = {
        "transformer_step_by_step.html": "Transformer, step by step",
        "transformer_guide.html": "Transformer guide", "build_llm_guide.html": "Build an LLM",
        "vit_guide.html": "ViT guide", "vlm_guide.html": "VLM guide",
        "diffusion_guide.html": "Diffusion guide", "mamba_guide.html": "Mamba guide",
        "moe_guide.html": "MoE guide", "bev_guide.html": "BEV guide", "vla_guide.html": "VLA guide",
    }.get(name, name)
    return nice + (f" #{frag}" if frag else "")


def render_readlist(entries):
    out = []
    items = []

    def flush():
        if items:
            out.append('<ul class="readlist">' + "".join(items) + "</ul>")
            items.clear()

    for e in entries:
        if e[0] == "label":
            flush()
            out.append(f'<p class="readlabel">{prose(e[1])}</p>')
        else:
            desc = f" <span>{prose(e[2])}</span>" if e[2] else ""
            items.append(f'<li><a href="{esc(e[1])}">{esc(link_label(e[1]))}</a>{desc}</li>')
    flush()
    return "\n".join(out)


# ===========================================================================
# labs/README.md: a small Markdown renderer and the lab table
# ===========================================================================
MD_INLINE = re.compile(r"(`[^`]+`)|\*\*(.+?)\*\*|\[([^\]]+)\]\(([^)\s]+)\)")


def md_inline(text, link_map=None):
    out, pos = [], 0
    for m in MD_INLINE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        if m.group(1):
            out.append(f"<code>{esc(m.group(1)[1:-1])}</code>")
        elif m.group(2) is not None:
            out.append(f"<b>{md_inline(m.group(2), link_map)}</b>")
        else:
            href = m.group(4)
            if link_map:
                href = link_map(href)
            out.append(f'<a href="{esc(href)}">{md_inline(m.group(3), link_map)}</a>')
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


def split_row(line):
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]


def md_blocks(lines, link_map=None, hooks=None):
    """Render Markdown lines (the subset used in labs/README.md) to HTML."""
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            out.append(f"<pre>{esc(chr(10).join(dedent(code)))}</pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = min(len(m.group(1)) + 1, 4)
            out.append(f"<h{lvl}>{md_inline(m.group(2), link_map)}</h{lvl}>")
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{3,}", lines[i + 1].strip()):
            head = split_row(s)
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            if hooks and "table" in hooks:
                rendered = hooks["table"](head, rows)
                if rendered is not None:
                    out.append(rendered)
                    i = j
                    continue
            t = ['<div class="table-scroll"><table><thead><tr>']
            t += [f"<th>{md_inline(h, link_map)}</th>" for h in head]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{md_inline(c, link_map)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            i = j
            continue
        m = re.match(r"^(\d+\.|[-*])\s+(.*)$", s)
        if m and indent_of(ln) == 0:
            ordered = m.group(1)[0].isdigit()
            items = []
            while i < len(lines):
                ln2 = lines[i]
                m2 = re.match(r"^(\d+\.|[-*])\s+(.*)$", ln2)
                if m2 and indent_of(ln2) == 0:
                    items.append([m2.group(2)])
                    i += 1
                    continue
                if ln2.strip() and indent_of(ln2) >= 2 and items:
                    items[-1].append(ln2)
                    i += 1
                    continue
                if not ln2.strip() and items:
                    k = i + 1
                    if k < len(lines) and lines[k].strip() and indent_of(lines[k]) >= 2:
                        items[-1].append("")
                        i += 1
                        continue
                break
            tag = "ol" if ordered else "ul"
            inner = []
            for it in items:
                body = [it[0]] + dedent(it[1:])
                simple = not any(not b.strip() or b.strip().startswith("```") for b in body)
                inner.append(md_inline(" ".join(b.strip() for b in body), link_map) if simple
                             else md_blocks(body, link_map))
            if hooks and ordered and "olist" in hooks:
                out.append(hooks["olist"](inner))
            else:
                out.append(f"<{tag}>" + "".join(f"<li>{x}</li>" for x in inner) + f"</{tag}>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("```", "#", "|")) \
                and not (re.match(r"^(\d+\.|[-*])\s+", lines[i]) and indent_of(lines[i]) == 0):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{md_inline(' '.join(para), link_map)}</p>")
    return "\n".join(out)


def parse_readme():
    text = (LABS / "README.md").read_text(encoding="utf-8")
    lines = text.split("\n")
    title = lines[0].lstrip("# ").strip()
    sections = []
    intro = []
    cur = None
    for ln in lines[1:]:
        if ln.startswith("## "):
            cur = {"head": ln[3:].strip(), "lines": []}
            sections.append(cur)
        elif cur is None:
            intro.append(ln)
        else:
            cur["lines"].append(ln)
    table = {}
    for ln in lines:
        if re.match(r"^\|\s*\[?0?\d", ln):
            cells = split_row(ln)
            num = int(re.search(r"\d+", cells[0]).group(0))
            table[num] = {"num": num, "what": cells[2], "read": cells[3], "time": cells[4]}
    return title, intro, sections, table


def first_link(md):
    m = re.search(r"\[([^\]]+)\]\(([^)\s]+)\)", md)
    return (m.group(1), m.group(2)) if m else (None, None)


# ===========================================================================
# Lab data
# ===========================================================================
def lab_titles():
    src = (LABS / "check.py").read_text(encoding="utf-8")
    return {int(n): (f, t) for n, f, t in re.findall(r'^lab\((\d+), "([^"]+)", "([^"]+)"\)', src, re.M)}


def load_lab(num, filename, short_title, readme_row):
    path = LABS / filename
    src = path.read_text(encoding="utf-8")
    if "raise NotImplementedError" not in src:
        print(f"warning: {filename} has no TODOs left; build the pages from the untouched skeletons "
              f"(git stash or git checkout labs/{filename})", file=sys.stderr)
    tree = ast.parse(src)
    doc = ast.get_docstring(tree, clean=False) or ""
    doc = inspect.cleandoc(doc)
    title, sections = parse_module_doc(doc)
    cells = labcells.split_cells(src)
    stem = Path(filename).stem
    m = re.match(r"Lab \d+:\s*(.*)$", title)
    long_title = m.group(1) if m else title
    long_title = long_title[:1].upper() + long_title[1:]
    run = section(sections, "Run")
    notes = []
    if run:
        for ln in run["lines"]:
            s = ln.strip()
            if s and not s.startswith(("Run:", "Hints are graded")):
                notes.append(s)
    read_text, read_href = first_link(readme_row["read"]) if readme_row else (None, None)
    return {
        "num": num, "file": filename, "stem": stem, "title": long_title, "short": short_title,
        "doc_title": title, "sections": sections, "cells": cells, "notes": notes,
        "time": readme_row["time"] if readme_row else "", "what": readme_row["what"] if readme_row else "",
        "read_md": readme_row["read"] if readme_row else "",
        "read_first": (read_text, read_href),
        "notebook": f"notebooks/{stem}.ipynb", "colab": COLAB.format(stem + ".ipynb"),
    }


def cell_groups(cells):
    """Group the code cells under their step banners: [(banner title, [cells])]."""
    groups = []
    title = "Setup: given code"
    cur = None
    for c in cells:
        if c["kind"] == "banner":
            title = c["title"]
            cur = None
            continue
        if c["kind"] != "code":
            continue
        if cur is None:
            cur = (title, [])
            groups.append(cur)
        cur[1].append(c)
    return groups


def hint_groups(cell):
    """[(function name, [items])] for the hint blocks of a cell."""
    out = []
    for h in cell.get("hints", []):
        out.append((h["func"], h["items"]))
    return out


def step_heading(title):
    m = re.match(r"^Step (\d+):\s*(.*)$", title)
    if m:
        return m.group(1), m.group(2), False
    m = re.match(r"^Given:\s*(.*)$", title)
    if m:
        return "G", "Given: " + m.group(1), True
    return "G", title, True


# ===========================================================================
# HTML pages
# ===========================================================================
def nav(lab, labs, bottom=False):
    num = lab["num"]
    prev_l = labs.get(num - 1)
    next_l = labs.get(num + 1)
    links = []
    if prev_l:
        links.append(f'<a href="lab{num - 1:02d}.html">← Lab {num - 1:02d}</a>')
    rt, rh = lab["read_first"]
    if rh:
        links.append(f'<a href="{esc(rh)}" title="Read this before the lab">Read first: {esc(rt)}</a>')
    if next_l:
        links.append(f'<a href="lab{num + 1:02d}.html">Lab {num + 1:02d}: {esc(next_l["short"])} →</a>')
    cls = "module-nav module-nav-bottom" if bottom else "module-nav"
    return (f'<nav class="{cls}" aria-label="Lab navigation">\n'
            f'    <a href="../index.html">⌂ Learning path</a>\n'
            f'    <span class="module-nav-step"><a href="index.html">Labs</a> · Lab {num:02d} of {len(labs)}</span>\n'
            f'    <span class="module-nav-links">{"".join(links)}</span>\n'
            f'</nav>')


def editor_html(cell_id, code, label, readonly=False):
    ro = " readonly" if readonly else ""
    return (f'<div class="editor"><pre class="hl" aria-hidden="true"><code></code></pre>'
            f'<textarea class="code" spellcheck="false" autocapitalize="off" autocomplete="off" '
            f'autocorrect="off" wrap="off" rows="{max(2, code.rstrip(chr(10)).count(chr(10)) + 1)}" '
            f'aria-label="{esc(label)}" data-cell="{esc(cell_id)}"{ro}>{esc(code.rstrip(chr(10)))}</textarea></div>')


def actions_html(scratch=False, solution=True, save=True):
    sol = '<button class="btn ghost" type="button" data-act="solution">Show solution</button>' if solution else ""
    save = '<button class="btn" type="button" data-act="save" disabled>Save</button>' if save and not scratch else ""
    return (f'<div class="cell-actions"><button class="btn primary" type="button" data-act="run" disabled>▶ Run</button>'
            f'{save}<span class="info" data-info></span><span class="sp"></span>{sol}</div>'
            f'<div class="solution" hidden></div><div class="output" hidden></div>')


def hints_html(cell):
    groups = hint_groups(cell)
    if not groups:
        return ""
    many = len(groups) > 1
    out = ['<div class="hints">']
    for func, items in groups:
        if many:
            out.append(f'<div class="hint-for">Hints for <code>{esc(func.rpartition(".")[2])}</code></div>')
        for k, it in enumerate(items):
            label = f"Hint {it['num']}" + (f" {it['label']}" if it["label"] else "")
            locked = " locked" if k > 0 else ""
            out.append(f'<details class="hint{locked}" data-hint="{k}"><summary>{esc(label)}</summary>'
                       f'<div class="hint-pre">{esc(it["text"])}</div></details>')
    out.append("</div>")
    return "".join(out)


def cell_doc_html(cell):
    parts = []
    if cell.get("doc"):
        parts.append(render_doc(cell["doc"]))
    for qual, sig, mdoc in cell.get("methods", []):
        parts.append(f'<h4>Write <code>{esc(sig.replace("self, ", "").replace("(self)", "()"))}</code></h4>')
        if mdoc:
            parts.append(render_doc(mdoc))
    return f'<div class="doc">{"".join(parts)}</div>' if parts else ""


def code_cell_html(cell):
    cid = cell["id"]
    todo = cell.get("todo")
    sig = cell.get("signature", cid)
    sig_disp = re.sub(r"^class ", "class ", sig)
    kind = "todo" if todo else "given"
    badge = '<span class="badge write" data-badge>to write</span>' if todo else \
        '<span class="badge given" data-badge>given</span>'
    head = (f'<div class="cell-head"><h3><code>{esc(sig_disp)}</code></h3>{badge}</div>')
    body = cell_doc_html(cell)
    label = f"Code for {cid}"
    if todo:
        inner = hints_html(cell) + editor_html(cid, cell["code"], label) + actions_html()
    else:
        inner = ('<details class="given-code"><summary>Show the given code (it is already written: '
                 'just run it)</summary>' + editor_html(cid, cell["code"], label) + '</details>'
                 + actions_html(solution=False, save=False))
    return (f'<article class="cell {kind}" id="cell-{esc(cid)}" data-cell="{esc(cid)}" data-kind="code">'
            f'{head}{body}{inner}</article>')


def header_cell_html(cell, num):
    extra = ""
    if "need" in cell["code"] and "import" in cell["code"]:
        extra = " Read the comment in it: this lab asks you to add imports here."
    return (f'<article class="cell given" id="cell-imports" data-cell="imports" data-kind="header">'
            f'<div class="cell-head"><h3>Imports</h3><span class="badge given" data-badge>run first</span></div>'
            f'<div class="doc"><p>The lab file\'s imports. Run this cell first (or press <b>Run all</b>).{extra}</p></div>'
            + editor_html("imports", cell["code"], "Imports") + actions_html(solution=False) + '</article>')


def lab_page(lab, labs):
    num = lab["num"]
    secs = lab["sections"]
    goal = section(secs, "Goal")
    maths = section(secs, "The maths")
    read = section(secs, "Read alongside")
    check_cmd = f"python labs/check.py {num}"
    notes = "".join(f"<span>{prose(n)}</span>" for n in lab["notes"])
    data = {
        "lab": num, "file": lab["file"], "title": lab["title"], "check": check_cmd,
        "notebook": lab["notebook"], "colab": lab["colab"], "solution": f"solutions/{lab['file']}",
        "cells": [{"id": c["id"], "kind": c["kind"], "todo": bool(c.get("todo"))}
                  for c in lab["cells"] if c["kind"] in ("code", "header")],
        "scratch": SCRATCH.get(num, "# Scratch cell\n"),
    }
    groups = cell_groups(lab["cells"])
    header_cell = next((c for c in lab["cells"] if c["kind"] == "header"), None)
    body = []
    if header_cell:
        body.append(f'<div class="step-group" id="imports"><h2><span class="num given">0</span> Imports</h2>'
                    f'{header_cell_html(header_cell, num)}</div>')
    for gi, (title, cells) in enumerate(groups):
        n, text, given = step_heading(title)
        gid = f"step-{n}" if n.isdigit() else f"group-{gi + 1}"
        num_cls = "num given" if given else "num"
        body.append(f'<div class="step-group" id="{gid}"><h2><span class="{num_cls}">{esc(n)}</span> '
                    f'{prose(text)}</h2>' + "".join(code_cell_html(c) for c in cells) + "</div>")
    scratch = (f'<div class="step-group" id="scratch"><h2><span class="num given">✎</span> Scratch cell</h2>'
               f'<article class="cell scratch" id="cell-scratch" data-cell="scratch" data-kind="scratch">'
               f'<div class="doc"><p>Experiment here: print shapes, call your functions on small inputs. '
               f'This cell runs in the same Python process as the cells above, is not part of <b>Run all</b> and '
               f'is not saved to the lab file (your browser keeps it).</p></div>'
               + editor_html("scratch", data["scratch"], "Scratch code") + actions_html(scratch=True, solution=False)
               + "</article></div>")
    body.append(scratch)
    toc = "".join(f'<li><a href="#{("step-" + step_heading(t)[0]) if step_heading(t)[0].isdigit() else "group-" + str(i + 1)}">'
                  f'{esc(step_heading(t)[1])}</a></li>' for i, (t, _) in enumerate(groups))
    read_html = render_readlist(parse_readlist(read["lines"])) if read else ""
    data_json = json.dumps(data).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lab {num:02d}: {esc(lab['short'])} · hands-on lab</title>
{FONTS}
<link rel="icon" href="data:,">
<link rel="stylesheet" href="assets/labs.css">
<script>document.documentElement.classList.add("js");</script>
</head>
<body data-page="lab" data-lab="{num}">
{nav(lab, labs)}
<div class="runbar" id="runbar">
    <div class="container runbar-inner">
        <span class="rb-status" id="rb-status" role="status"><span class="dot"></span><span class="txt">Looking for the lab runner…</span></span>
        <span class="rb-actions">
            <button class="btn primary" type="button" id="btn-runall" disabled title="Run every cell from the top">▶ Run all</button>
            <button class="btn green" type="button" id="btn-check" disabled title="Save, then run python labs/check.py {num}">✓ Run checks</button>
            <button class="btn" type="button" id="btn-reset" disabled title="Restart this lab's Python process">↺ Reset<span class="long"> kernel</span></button>
            <button class="btn" type="button" id="btn-save" disabled title="Save all cells to labs/{esc(lab['file'])} (Ctrl+S)">Save</button>
            <span class="saved" id="rb-saved" aria-live="polite"></span>
        </span>
    </div>
    <div class="check-panel" id="check-panel" hidden></div>
</div>

<header class="page-head">
    <div class="container">
        <p class="kicker">Hands-on lab {num:02d} of {len(labs)} · {esc(lab['short'])}</p>
        <h1>{esc(lab['title'])}</h1>
        {render_goal(goal['lines']) if goal else ''}
        <div class="meta"><span><b>Time:</b> {esc(lab['time'])}</span><span><b>Check:</b> <span class="cmd">{esc(check_cmd)}</span></span>{notes}</div>
        <div class="pill-links">
            <a class="pill colab" href="{esc(lab['colab'])}" target="_blank" rel="noopener">Open in Colab ↗</a>
            <a class="pill" href="{esc(lab['notebook'])}">Notebook (.ipynb)</a>
            <a class="pill" href="{esc(lab['file'])}">Lab file (.py)</a>
            <a class="pill" href="index.html">All labs</a>
        </div>
    </div>
</header>

<main class="container">
    <div id="mode-banner"></div>
    <section class="block maths" id="maths">
        <h2>{prose(maths['head']) if maths else 'The maths, step by step'}</h2>
        {render_maths(maths['lines']) if maths else ''}
    </section>
    <section class="block" id="read">
        <h2>Read alongside</h2>
        {read_html}
    </section>
    <section class="block" id="how">
        <h2>How to work on this page</h2>
        <ol class="lead-list">
            <li>Write each function in its cell, in order. The hints open one at a time: try Hint 1 first.</li>
            <li><b>▶ Run</b> (<kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>Enter</kbd>) defines the cell in this lab's Python process; variables persist between cells, as in a notebook. Try your functions in the scratch cell at the bottom.</li>
            <li><b>✓ Run checks</b> saves the file and runs <span class="cmd">{esc(check_cmd)}</span>: each step turns from TODO to PASS. A FAIL shows what was expected.</li>
            <li><b>Save</b> (<kbd>Ctrl</kbd>/<kbd>⌘</kbd>+<kbd>S</kbd>) writes all cells to <span class="cmd">labs/{esc(lab['file'])}</span> (the previous version is kept in <span class="cmd">labs/.backup/</span>), so you can also continue in any editor. After <b>↺ Reset kernel</b>, press <b>▶ Run all</b>.</li>
        </ol>
        <p>Editor keys: <kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd> indent and dedent by 4 spaces, <kbd>Enter</kbd> keeps the indentation, <kbd>Shift</kbd>+<kbd>Enter</kbd> runs the cell and moves to the next one.</p>
        <p class="readlabel">Cells on this page:</p>
        <ul class="readlist">{toc}<li><a href="#scratch">Scratch cell</a></li></ul>
    </section>
    {''.join(body)}
</main>

{nav(lab, labs, bottom=True)}
<footer class="page-foot"><p>Generated from <a href="{esc(lab['file'])}">labs/{esc(lab['file'])}</a> by <span class="cmd">labs/tools/build_pages.py</span> · <a href="index.html">all labs</a> · <a href="../index.html">learning path</a></p></footer>
<script type="application/json" id="lab-data">{data_json}</script>
<script src="assets/labs.js"></script>
</body>
</html>
"""


def hub_page(labs, readme):
    title, intro, sections, table = readme
    intro_html = md_blocks([ln for ln in intro if ln.strip() or True])

    def link_map(href):
        m = re.fullmatch(r"(lab\d\d)_[a-z0-9_]+\.py", href)
        return f"{m.group(1)}.html" if m else href

    def table_hook(head, rows):
        if not head or head[0].lower() != "lab":
            return None
        cards = []
        for lab in labs.values():
            n = lab["num"]
            rt, rh = lab["read_first"]
            cards.append(
                f'<li class="labcard" id="card-{n}" data-lab="{n}">'
                f'<div class="lnum">{n:02d}</div><div class="lbody">'
                f'<h3><a href="lab{n:02d}.html">{esc(lab["short"])}</a></h3>'
                f'<p class="what">{md_inline(lab["what"], link_map)}</p>'
                f'<p class="readwith"><b>Read with:</b> {md_inline(lab["read_md"], link_map)}</p>'
                f'<div class="row"><span>⏱ {esc(lab["time"])}</span>'
                f'<a class="open" href="lab{n:02d}.html">Open lab →</a>'
                f'<a href="{esc(lab["notebook"])}">Notebook</a>'
                f'<a href="{esc(lab["colab"])}" target="_blank" rel="noopener">Colab ↗</a>'
                f'<a href="{esc(lab["file"])}">.py</a></div>'
                f'<div class="status" data-status><span class="badge" data-badge>not checked</span>'
                f'<span class="bar"><span></span></span><span data-count></span></div>'
                f'</div></li>')
        return ('<div class="hubbar" id="hubbar"><span class="rb-status" id="rb-status" role="status">'
                '<span class="dot"></span><span class="txt">Looking for the lab runner…</span></span>'
                '<span class="sp"></span><span class="saved" id="hub-summary"></span>'
                '<button class="btn green" type="button" id="btn-checkall" disabled>✓ Check all labs</button></div>'
                '<ol class="labgrid">' + "".join(cards) + "</ol>")

    def ways_hook(items):
        cards = "".join(f'<div class="way"><span class="n">{i + 1}</span><div>{x}</div></div>'
                        for i, x in enumerate(items))
        return f'<div class="ways">{cards}</div>'

    # the lab cards first, then how to work; the rest in README order
    sections = sorted(sections, key=lambda s: 0 if s["head"].lower().startswith("recommended")
                      else 1 if s["head"].lower().startswith("three ways") else 2)
    sec_html = []
    for s in sections:
        sid = re.sub(r"[^a-z0-9]+", "-", s["head"].lower()).strip("-")
        hooks = {"table": table_hook}
        if s["head"].lower().startswith("three ways"):
            hooks["olist"] = ways_hook
        content = md_blocks(s["lines"], link_map, hooks=hooks)
        head = "The ten labs, in order" if sid == "recommended-order" else md_inline(s["head"])
        sec_html.append(f'<section class="block md" id="{sid}"><h2>{head}</h2>{content}</section>')
    first = labs[min(labs)]
    toc = "".join(f'<a class="pill" href="#{re.sub(r"[^a-z0-9]+", "-", s["head"].lower()).strip("-")}">'
                  f'{"The ten labs" if s["head"].lower().startswith("recommended") else esc(s["head"])}</a>'
                  for s in sections)
    data = {"labs": [{"num": lab["num"], "file": lab["file"]} for lab in labs.values()]}
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hands-on labs · Model Study</title>
{FONTS}
<link rel="icon" href="data:,">
<link rel="stylesheet" href="assets/labs.css">
<script>document.documentElement.classList.add("js");</script>
</head>
<body data-page="hub">
<nav class="module-nav" aria-label="Lab navigation">
    <a href="../index.html">⌂ Learning path</a>
    <span class="module-nav-step">Labs · {len(labs)} hands-on labs</span>
    <span class="module-nav-links"><a href="#three-ways-to-work">How to work</a><a href="lab01.html">Start with lab 01: {esc(first['short'])} →</a></span>
</nav>
<header class="page-head">
    <div class="container">
        <p class="kicker">Hands-on track</p>
        <h1>{md_inline(title)}</h1>
        <div class="md subtitle-block">{intro_html}</div>
        <div class="pill-links">{toc}</div>
    </div>
</header>
<main class="container">
    <div id="mode-banner"></div>
    {''.join(sec_html)}
</main>
<nav class="module-nav module-nav-bottom" aria-label="Lab navigation">
    <a href="../index.html">⌂ Learning path</a>
    <span class="module-nav-step">Labs</span>
    <span class="module-nav-links"><a href="lab01.html">Lab 01 →</a></span>
</nav>
<footer class="page-foot"><p>Generated from <a href="README.md">labs/README.md</a> by <span class="cmd">labs/tools/build_pages.py</span> · <a href="../index.html">learning path</a></p></footer>
<script type="application/json" id="hub-data">{json.dumps(data)}</script>
<script src="assets/labs.js"></script>
</body>
</html>
"""


# ===========================================================================
# Notebooks (nbformat 4, written directly)
# ===========================================================================
def nb_lines(text):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]] if lines else []


def md_cell(cid, text):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": nb_lines(text.strip("\n"))}


def code_cell(cid, code, meta=None):
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": meta or {},
            "outputs": [], "source": nb_lines(code.rstrip("\n"))}


def nb_rel(href):
    """Links in the lab docstrings are relative to labs/; the notebooks live in labs/notebooks/."""
    return href if re.match(r"^[a-z]+:", href) else "../" + href


SETUP = f'''# Setup: make the checker (labs/check.py) importable. Run this cell first.
# In a clone of the repository (Jupyter, VS Code) it uses labs/check.py;
# elsewhere (e.g. Google Colab) it downloads check.py, which needs only torch.
import pathlib
import sys
import urllib.request


def _find_labs_dir():
    here = pathlib.Path.cwd().resolve()
    for d in [here, *here.parents]:
        for cand in (d, d / "labs"):
            if (cand / "check.py").is_file() and (cand / "solutions").is_dir():
                return cand
    return None


LABS_DIR = _find_labs_dir()
if LABS_DIR is None:
    LABS_DIR = pathlib.Path("hc_model_study_labs").resolve()
    LABS_DIR.mkdir(exist_ok=True)
    if not (LABS_DIR / "check.py").is_file():
        urllib.request.urlretrieve("{RAW_CHECK}", LABS_DIR / "check.py")
sys.path.insert(0, str(LABS_DIR))
import check  # noqa: E402

import torch  # noqa: E402
print("checker:", LABS_DIR / "check.py", "| torch", torch.__version__)'''


def notebook(lab):
    num = lab["num"]
    secs = lab["sections"]
    goal = section(secs, "Goal")
    maths = section(secs, "The maths")
    read = section(secs, "Read alongside")
    pre = f"lab{num:02d}"
    cells = []
    md = [f"# Lab {num:02d}: {lab['title']}", "",
          f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({lab['colab']})"
          f" · lab page: [`labs/lab{num:02d}.html`](../lab{num:02d}.html) · file: [`labs/{lab['file']}`](../{lab['file']})",
          "", f"**Time:** {lab['time']}"]
    if lab["notes"]:
        md += ["", " ".join(lab["notes"])]
    if goal:
        md += ["", "## Goal", "", "\n".join(ln.strip() if not ln.lstrip().startswith("* ") else "- " + ln.lstrip()[2:]
                                           for ln in goal["lines"])]
    if maths:
        md += ["", "## The maths, step by step", "", "```text", "\n".join(dedent(maths["lines"])), "```"]
    if read:
        md += ["", "## Read alongside", ""]
        for e in parse_readlist(read["lines"]):
            if e[0] == "label":
                md.append(f"*{e[1]}*")
            else:
                md.append(f"- [{link_label(e[1])}]({nb_rel(e[1])}){': ' + e[2] if e[2] else ''}")
    md += ["", "## How to work", "",
           "1. Run the setup cell and the imports cell.",
           "2. Implement the functions in order, replacing `raise NotImplementedError  # TODO`. "
           "The hints above each cell open one at a time; try Hint 1 first.",
           "3. Run each cell after editing it, then run the check cell at the end: "
           "`check.run_lab(%d, namespace=globals())` checks the functions defined in this notebook "
           "(TODO = not started, PASS = matches the reference, FAIL = read the message)." % num,
           "",
           "Your work stays in this notebook; it does not change `labs/%s` or the lab page. "
           "The reference solution is `labs/solutions/%s`." % (lab["file"], lab["file"])]
    cells.append(md_cell(f"{pre}-intro", "\n".join(md)))
    cells.append(code_cell(f"{pre}-setup", SETUP, {"hc_lab": {"cell": "setup"}}))
    header = next((c for c in lab["cells"] if c["kind"] == "header"), None)
    if header:
        cells.append(md_cell(f"{pre}-imports-md", "### Imports"))
        cells.append(code_cell(f"{pre}-imports", header["code"], {"hc_lab": {"cell": "imports"}}))
    for title, group in cell_groups(lab["cells"]):
        n, text, given = step_heading(title)
        head = f"## Step {n}: {text}" if n.isdigit() else f"## {text}"
        cells.append(md_cell(f"{pre}-{len(cells)}-h", head))
        for c in group:
            parts = []
            if not c.get("todo"):
                parts.append(f"Given: `{c['signature']}` is already written. Run the cell.")
            for func, items in hint_groups(c):
                parts.append(f"**Hints for `{func}`** (open one at a time):")
                for it in items:
                    label = f"Hint {it['num']}" + (f" {it['label']}" if it["label"] else "")
                    parts.append(f"<details><summary>{esc(label)}</summary>\n\n```text\n{it['text']}\n```\n\n</details>")
            if parts:
                cells.append(md_cell(f"{pre}-{c['id']}-md".replace("_", "-")[:64], "\n\n".join(parts)))
            cells.append(code_cell(f"{pre}-{c['id']}".replace("_", "-")[:64], c["code"],
                                   {"hc_lab": {"cell": c["id"]}}))
    cells.append(md_cell(f"{pre}-check-md",
                         "## Check\n\nRun this after each function you finish. Steps you have not started show "
                         "TODO; finished steps show PASS. For a full traceback of a failing step: "
                         f"`check.run_lab({num}, globals(), verbose=True)`."))
    cells.append(code_cell(f"{pre}-check", f"check.run_lab({num}, namespace=globals());",
                           {"hc_lab": {"cell": "check"}}))
    cells.append(md_cell(f"{pre}-scratch-md",
                         "## Scratch\n\nTry your functions here: print shapes, call them on small inputs. "
                         "(Functions you have not written yet raise `NotImplementedError`.)"))
    cells.append(code_cell(f"{pre}-scratch", SCRATCH.get(num, ""), {"hc_lab": {"cell": "scratch"}}))
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"name": f"{lab['stem']}.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


# ===========================================================================
def build():
    titles = lab_titles()
    readme = parse_readme()
    table = readme[3]
    labs = {}
    for num, (filename, short) in sorted(titles.items()):
        labs[num] = load_lab(num, filename, short, table.get(num))
    outputs = {LABS / "index.html": hub_page(labs, readme)}
    for num, lab in labs.items():
        outputs[LABS / f"lab{num:02d}.html"] = lab_page(lab, labs)
        outputs[LABS / "notebooks" / f"{lab['stem']}.ipynb"] = notebook(lab)
    return outputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="only report files that are out of date")
    args = p.parse_args(argv)
    outputs = build()
    stale = []
    for path, text in outputs.items():
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old != text:
            stale.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    rel = [str(p.relative_to(LABS.parent)) for p in stale]
    if args.check:
        if stale:
            print("out of date (run python labs/tools/build_pages.py):\n  " + "\n  ".join(rel))
            return 1
        print(f"all {len(outputs)} generated files are up to date")
        return 0
    print(f"wrote {len(stale)} of {len(outputs)} files" + (":\n  " + "\n  ".join(rel) if rel else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
