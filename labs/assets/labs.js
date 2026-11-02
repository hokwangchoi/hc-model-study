/* Lab hub and lab pages: a small notebook-like client for labs/serve.py.
 *
 * Without the runner (opened as a file, or from a plain static server) the
 * pages stay readable: editors are read-only and a banner explains how to
 * start `python labs/serve.py`. With the runner, the editors load the CURRENT
 * labs/labNN_*.py, and Run / Save / Run checks talk to the local API with the
 * token from the URL fragment (#token=...). No external libraries. */
(function () {
    "use strict";

    // ------------------------------------------------------------------ utilities
    const $ = (sel, root) => (root || document).querySelector(sel);
    const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
    const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    function el(tag, attrs, html) {
        const e = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs || {})) {
            if (k === "class") e.className = v; else if (v !== null && v !== undefined) e.setAttribute(k, v);
        }
        if (html !== undefined) e.innerHTML = html;
        return e;
    }
    const store = {
        get(k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
        set(k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
    };
    const TOKEN_KEY = "hc-labs-token:" + location.host;
    let memToken = null;

    function takeToken() {
        const m = /(?:^#|&)token=([A-Za-z0-9_\-]+)/.exec(location.hash);
        if (m) {
            memToken = m[1];
            store.set(TOKEN_KEY, memToken);
            // keep the token out of the address bar (and out of copied links)
            try { history.replaceState(null, "", location.pathname + location.search); } catch (e) { /* ignore */ }
        }
        return memToken || store.get(TOKEN_KEY);
    }

    async function api(method, path, body) {
        const headers = { "X-Lab-Token": takeToken() || "" };
        const opts = { method, headers, cache: "no-store" };
        if (body !== undefined) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
        let resp;
        try { resp = await fetch(path, opts); } catch (e) { return { ok: false, status: 0, data: null, error: String(e) }; }
        let data = null;
        if ((resp.headers.get("Content-Type") || "").includes("application/json")) {
            try { data = await resp.json(); } catch (e) { data = null; }
        }
        return { ok: resp.ok && data !== null, status: resp.status, data };
    }

    async function detectRunner() {
        if (location.protocol === "file:") return { mode: "file" };
        if (!/^(127\.0\.0\.1|localhost)$/.test(location.hostname)) return { mode: "noserver" };
        const r = await api("GET", "/api/status");
        if (r.ok && r.data && r.data.ok) return { mode: "connected", info: r.data };
        if (r.status === 403 && r.data) return { mode: "badtoken", error: r.data.error };
        return { mode: "noserver" };
    }

    function setStatus(kind, text) {
        const box = $("#rb-status");
        if (!box) return;
        $(".dot", box).className = "dot" + (kind ? " " + kind : "");
        $(".txt", box).textContent = text;
    }

    function fitNav() {
        const nav = $(".module-nav");
        if (!nav) return;
        const sticky = getComputedStyle(nav).position === "sticky";
        document.documentElement.style.setProperty("--modnav-h", (sticky ? nav.offsetHeight : 0) + "px");
    }

    function banner(mode, ctx) {
        const box = $("#mode-banner");
        if (!box) return;
        const isLab = !!ctx.lab;
        const start = '<pre>python labs/serve.py --open</pre>';
        let html = "";
        if (mode === "file" || mode === "noserver") {
            const where = mode === "file" ? "opened as a file" : "served without the lab runner";
            html = `<h3>Read-only: this page is ${where}</h3>
                <p>Everything is here to read. To ${isLab ? "write and run code on this page" : "run the labs in your browser"}, start the local lab runner from the repository root (Python 3 with PyTorch 2.1 or newer):</p>
                ${start}
                <p>It prints a URL like <code>http://127.0.0.1:8765/labs/index.html#token=…</code>; open it (<code>--open</code> does that for you). Other ways to work:</p>
                <ul>
                ${isLab ? `<li>Jupyter or VS Code: open <a href="${esc(ctx.lab.notebook)}">${esc(ctx.lab.notebook.split("/").pop())}</a> (<code>pip install notebook</code>, or VS Code with <code>pip install ipykernel</code>).</li>
                <li><a href="${esc(ctx.lab.colab)}" target="_blank" rel="noopener">Open in Colab ↗</a> (works once the notebooks are on GitHub).</li>
                <li>Any editor: edit <a href="${esc(ctx.lab.file)}">labs/${esc(ctx.lab.file)}</a> and run <code>${esc(ctx.lab.check)}</code>.</li>`
                    : `<li>Jupyter or VS Code: the notebooks in <a href="notebooks/">labs/notebooks/</a>.</li>
                <li>Google Colab: the "Colab" link on each lab card.</li>
                <li>Any editor: edit <code>labs/labNN_*.py</code> and run <code>python labs/check.py N</code>.</li>`}
                </ul>`;
        } else if (mode === "badtoken") {
            html = `<h3>The lab runner is running, but this tab has no valid token</h3>
                <p>Open the URL that <code>python labs/serve.py</code> printed in its terminal (it ends with <code>#token=…</code>). The token changes every time the runner starts. Until then this page is read-only.</p>`;
        } else if (mode === "offline") {
            html = `<h3>The lab runner stopped</h3>
                <p>Start it again with <code>python labs/serve.py --open</code> and open the new URL. Unsaved edits in this tab are still here: copy them first if you need them.</p>`;
        }
        box.className = html ? "banner" + (mode === "offline" ? " err" : "") : "";
        box.innerHTML = html;
    }

    // ------------------------------------------------------------------ Python highlighting
    const KW = new Set(("False None True and as assert async await break class continue def del elif else except " +
        "finally for from global if import in is lambda nonlocal not or pass raise return try while with yield").split(" "));
    const BI = new Set(("abs all any bool dict enumerate float getattr hasattr int isinstance len list map max min " +
        "print range repr reversed round set sorted str sum super tuple type zip NotImplementedError ValueError " +
        "TypeError Exception").split(" "));
    const TOKEN_RE = /(#[^\n]*)|([rRbBuUfF]{0,2}"""[\s\S]*?(?:"""|$)|[rRbBuUfF]{0,2}'''[\s\S]*?(?:'''|$))|([rRbBuUfF]{0,2}"(?:\\.|[^"\\\n])*"?|[rRbBuUfF]{0,2}'(?:\\.|[^'\\\n])*'?)|(\b\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?j?\b)|(@[A-Za-z_][\w.]*)|([A-Za-z_]\w*)/g;

    function highlight(src) {
        let out = "", pos = 0, prevWord = "", m;
        TOKEN_RE.lastIndex = 0;
        while ((m = TOKEN_RE.exec(src)) !== null) {
            if (m[0] === "") { TOKEN_RE.lastIndex++; continue; }
            out += esc(src.slice(pos, m.index));
            const t = esc(m[0]);
            if (m[1]) out += `<span class="cm">${t.replace(/\bTODO\b/, '<span class="todo">TODO</span>')}</span>`;
            else if (m[2]) out += `<span class="ds">${t}</span>`;
            else if (m[3]) out += `<span class="str">${t}</span>`;
            else if (m[4]) out += `<span class="num">${t}</span>`;
            else if (m[5]) out += `<span class="dec">${t}</span>`;
            else {
                const w = m[6];
                if (KW.has(w)) out += `<span class="kw">${t}</span>`;
                else if (prevWord === "def" || prevWord === "class") out += `<span class="fn">${t}</span>`;
                else if (w === "self") out += `<span class="self">${t}</span>`;
                else if (BI.has(w)) out += `<span class="bi">${t}</span>`;
                else out += t;
                prevWord = w;
                pos = TOKEN_RE.lastIndex;
                continue;
            }
            prevWord = "";
            pos = TOKEN_RE.lastIndex;
        }
        return out + esc(src.slice(pos));
    }

    // ------------------------------------------------------------------ editor
    class Editor {
        constructor(ta, opts) {
            this.ta = ta;
            this.pre = ta.previousElementSibling;
            this.code = this.pre.firstElementChild;
            this.opts = opts || {};
            ta.rows = 1;              // the height follows the content (rows only matter without JS)
            ta.addEventListener("input", () => { this.render(); if (this.opts.onChange) this.opts.onChange(); });
            ta.addEventListener("scroll", () => this.sync());
            ta.addEventListener("keydown", (e) => this.onKey(e));
            this.render();
        }
        get value() { return this.ta.value; }
        set value(v) { this.ta.value = v; this.render(); }
        render() {
            this.code.innerHTML = highlight(this.ta.value) + "\n";
            this.autosize();
            this.sync();
        }
        autosize() {
            const ta = this.ta;
            if (!ta.offsetParent) { this.pendingSize = true; return; }   // hidden (collapsed): size it when shown
            this.pendingSize = false;
            ta.style.height = "auto";
            const extra = ta.offsetHeight - ta.clientHeight;
            ta.style.height = (ta.scrollHeight + extra) + "px";
        }
        sync() { this.pre.scrollLeft = this.ta.scrollLeft; this.pre.scrollTop = this.ta.scrollTop; }
        insert(text) {
            const ta = this.ta;
            ta.focus();
            let done = false;
            try { done = document.execCommand("insertText", false, text); } catch (e) { done = false; }
            if (!done) {
                ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, "end");
                ta.dispatchEvent(new Event("input"));
            }
        }
        lineRange() {
            const v = this.ta.value;
            const s = this.ta.selectionStart;
            let e = this.ta.selectionEnd;
            if (e > s && v[e - 1] === "\n") e--;
            const ls = v.lastIndexOf("\n", s - 1) + 1;
            let le = v.indexOf("\n", e);
            if (le < 0) le = v.length;
            return [ls, le];
        }
        mapLines(fn) {
            const ta = this.ta;
            const [ls, le] = this.lineRange();
            const block = ta.value.slice(ls, le);
            const out = block.split("\n").map(fn).join("\n");
            ta.setSelectionRange(ls, le);
            this.insert(out);
            ta.setSelectionRange(ls, ls + out.length);
        }
        onKey(e) {
            const ta = this.ta;
            const mod = e.ctrlKey || e.metaKey;
            if (mod && e.key === "Enter") { e.preventDefault(); if (this.opts.onRun) this.opts.onRun(); return; }
            if (e.shiftKey && e.key === "Enter" && !mod && !e.altKey) { e.preventDefault(); if (this.opts.onRunNext) this.opts.onRunNext(); return; }
            if (e.key === "Escape") { if (this.opts.onEscape) this.opts.onEscape(); return; }
            if (ta.readOnly || e.isComposing) return;
            if (e.key === "Tab" && !mod && !e.altKey) {
                e.preventDefault();
                const multi = ta.value.slice(ta.selectionStart, ta.selectionEnd).includes("\n");
                if (e.shiftKey) this.mapLines((ln) => ln.replace(/^ {1,4}/, ""));
                else if (multi) this.mapLines((ln) => (ln.trim() ? "    " + ln : ln));
                else {
                    const ls = ta.value.lastIndexOf("\n", ta.selectionStart - 1) + 1;
                    const col = ta.selectionStart - ls;
                    this.insert(" ".repeat(4 - (col % 4)));
                }
                return;
            }
            if (e.key === "Enter" && !mod && !e.altKey && !e.shiftKey) {
                e.preventDefault();
                const v = ta.value, s = ta.selectionStart;
                const line = v.slice(v.lastIndexOf("\n", s - 1) + 1, s);
                let ind = (/^ */.exec(line) || [""])[0];
                const code = line.replace(/#.*$/, "").trimEnd();
                if (code.endsWith(":")) ind += "    ";
                else if (/^\s*(return|pass|raise|break|continue)\b/.test(line) && ind.length >= 4) ind = ind.slice(4);
                this.insert("\n" + ind);
                return;
            }
            if (e.key === "Backspace" && !mod && !e.altKey && ta.selectionStart === ta.selectionEnd) {
                const v = ta.value, s = ta.selectionStart;
                const before = v.slice(v.lastIndexOf("\n", s - 1) + 1, s);
                if (before.length >= 2 && /^ +$/.test(before)) {
                    e.preventDefault();
                    const n = before.length % 4 || 4;
                    ta.setSelectionRange(s - n, s);
                    let done = false;
                    try { done = document.execCommand("delete"); } catch (err) { done = false; }
                    if (!done) { ta.setRangeText("", s - n, s, "end"); ta.dispatchEvent(new Event("input")); }
                }
            }
        }
    }

    // ------------------------------------------------------------------ progressive hints
    function initHints(root) {
        $$("details.hint", root).forEach((d) => {
            d.addEventListener("toggle", () => {
                if (!d.open) return;
                let n = d.nextElementSibling;
                if (n && n.matches("details.hint.locked")) n.classList.remove("locked");
            });
        });
    }

    // ================================================================== lab page
    function labPage() {
        const LAB = JSON.parse($("#lab-data").textContent);
        const N = LAB.lab;
        const cells = new Map();
        const order = [];
        const state = { mode: "unknown", base: null, layout: null, busy: 0, solution: null, queue: Promise.resolve(), runs: 0 };

        $$("article.cell").forEach((art) => {
            const id = art.dataset.cell;
            const ta = $("textarea.code", art);
            const c = {
                id, art, kind: art.dataset.kind, ta, skeleton: ta.value, saved: ta.value, sep: "\n\n",
                info: $("[data-info]", art), out: $(".output", art), sol: $(".solution", art),
                badge: $("[data-badge]", art), details: $("details.given-code", art),
            };
            c.editor = new Editor(ta, {
                onChange: () => { if (c.kind === "scratch") saveScratch(); else updateDirty(); },
                onRun: () => runCell(c),
                onRunNext: () => runCell(c).then(() => focusNext(c)),
                onEscape: () => { const b = $('[data-act="run"]', art); if (b) b.focus(); },
            });
            if (c.details) c.details.addEventListener("toggle", () => { if (c.details.open && c.editor.pendingSize) c.editor.autosize(); });
            $$("[data-act]", art).forEach((b) => b.addEventListener("click", () => {
                const act = b.dataset.act;
                if (act === "run") runCell(c);
                else if (act === "save") save();
                else if (act === "solution") toggleSolution(c);
            }));
            cells.set(id, c);
            if (c.kind !== "scratch") order.push(c);
        });
        const scratch = cells.get("scratch");
        const SCRATCH_KEY = `hc-labs-scratch:${N}`;
        if (scratch) {
            const s = store.get(SCRATCH_KEY);
            if (s !== null) scratch.editor.value = s;
        }
        let scratchTimer = null;
        function saveScratch() {
            clearTimeout(scratchTimer);
            scratchTimer = setTimeout(() => store.set(SCRATCH_KEY, scratch.editor.value), 300);
        }
        initHints(document);

        function focusNext(c) {
            const list = order.concat(scratch ? [scratch] : []);
            const i = list.indexOf(c);
            const next = list[i + 1];
            if (!next) return;
            if (next.details && !next.details.open) next.details.open = true;
            next.ta.focus();
            next.art.scrollIntoView({ block: "nearest" });
        }

        // ---- buttons and global keys
        const btn = { runall: $("#btn-runall"), check: $("#btn-check"), reset: $("#btn-reset"), save: $("#btn-save") };
        btn.runall.addEventListener("click", runAll);
        btn.check.addEventListener("click", runChecks);
        btn.reset.addEventListener("click", resetKernel);
        btn.save.addEventListener("click", () => save());
        document.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
                e.preventDefault();
                if (state.mode === "connected") save();
            }
        });
        window.addEventListener("beforeunload", (e) => {
            if (dirtyCells().length && state.mode === "connected") { e.preventDefault(); e.returnValue = ""; }
        });

        function enable(on) {
            Object.values(btn).forEach((b) => { b.disabled = !on; });
            $$('[data-act="run"], [data-act="save"]').forEach((b) => { b.disabled = !on; });
            cells.forEach((c) => { c.ta.readOnly = !on && c.kind !== "scratch" ? true : !on; });
        }

        function dirtyCells() { return order.filter((c) => c.editor.value !== c.saved); }
        function updateDirty() {
            const d = dirtyCells();
            const s = $("#rb-saved");
            order.forEach((c) => { if (c.info && !c.info.dataset.sticky) c.info.textContent = c.editor.value !== c.saved ? "edited" : c.info.dataset.last || ""; });
            if (state.mode !== "connected") { s.textContent = ""; return; }
            s.className = "saved" + (d.length ? " dirty" : "");
            s.textContent = d.length ? `Unsaved changes (${d.length} cell${d.length > 1 ? "s" : ""})` : (state.savedAt ? `Saved ${state.savedAt}` : "All changes saved");
        }

        function busy(delta, text) {
            state.busy += delta;
            if (state.mode !== "connected") return;
            if (state.busy > 0) setStatus("busy", text || "Running…");
            else setStatus("ok", state.idleText);
        }

        // ---- loading the current file
        async function loadCells() {
            const r = await api("GET", `/api/lab/${N}/cells`);
            if (!r.ok) { setStatus("off", "Could not load the lab file: " + ((r.data && r.data.error) || r.status)); return false; }
            const d = r.data;
            state.base = d.hash;
            const layout = [];
            const seen = new Set();
            let prev = null;
            for (const sc of d.cells) {
                if (sc.kind === "code" || sc.kind === "header") {
                    const c = cells.get(sc.id);
                    const code = sc.code.replace(/\n+$/, "");
                    if (c && c.kind !== "scratch" && !seen.has(sc.id)) {
                        c.editor.value = code; c.sep = sc.sep; c.saved = code;
                        layout.push({ type: "cell", id: sc.id });
                        seen.add(sc.id);
                        prev = c;
                    } else if (prev) {             // a def the page does not know: keep it in the cell above
                        const merged = prev.editor.value + "\n" + prev.sep + code;
                        prev.editor.value = merged; prev.saved = merged; prev.sep = sc.sep;
                    } else {
                        layout.push({ type: "raw", text: code + "\n" + sc.sep });
                    }
                } else {
                    layout.push({ type: "raw", text: sc.source });
                }
            }
            // cells of the page missing from the file: put the skeleton back after the previous cell
            let restored = 0;
            order.forEach((c, i) => {
                if (seen.has(c.id)) return;
                let at = -1;
                for (let j = i - 1; j >= 0 && at < 0; j--) at = layout.findIndex((x) => x.type === "cell" && x.id === order[j].id);
                if (at < 0) at = layout.findIndex((x) => x.type === "raw") ;
                layout.splice(at + 1, 0, { type: "cell", id: c.id });
                c.editor.value = c.skeleton; c.saved = ""; c.sep = "\n\n";
                if (c.info) { c.info.textContent = "not in your file: restored from the skeleton"; c.info.dataset.sticky = "1"; }
                seen.add(c.id);
                restored++;
            });
            state.layout = layout;
            if (d.syntax_error) {
                const b = $("#mode-banner");
                b.className = "banner err";
                b.innerHTML = `<h3>Your lab file has a syntax error</h3><p><code>${esc(d.syntax_error)}</code> in <code>labs/${esc(LAB.file)}</code>. The cells below are loaded from it; fix the error in its cell and save.</p>`;
            }
            updateDirty();
            return true;
        }

        function parts() {
            return state.layout.map((x) => x.type === "raw" ? { kind: "raw", text: x.text }
                : { kind: "cell", id: x.id, code: cells.get(x.id).editor.value, sep: cells.get(x.id).sep });
        }

        function markSaved(res) {
            state.base = res.hash;
            order.forEach((c) => { c.saved = c.editor.value; if (c.info && c.info.dataset.sticky) { delete c.info.dataset.sticky; c.info.textContent = ""; } });
            const t = new Date();
            state.savedAt = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
            updateDirty();
            const s = $("#rb-saved");
            if (res.syntax_error) { s.className = "saved dirty"; s.textContent = `Saved, but ${res.syntax_error}`; }
            if (res.hints_dropped && res.hints_dropped.length) s.title = "Hints not kept for: " + res.hints_dropped.join(", ");
        }

        async function postWithConflict(path, body) {
            let r = await api("POST", path, body);
            if (r.status === 409 && r.data) {
                const ok = window.confirm(r.data.error + "\n\nOverwrite the file with the code on this page? " +
                    "(The current file is copied to labs/.backup/ first.)\n\nCancel keeps the file; reload the page to load it.");
                if (!ok) return null;
                r = await api("POST", path, Object.assign({}, body, { force: true }));
            }
            return r;
        }

        async function save() {
            if (state.mode !== "connected" || !state.layout) return false;
            const s = $("#rb-saved");
            s.className = "saved"; s.textContent = "Saving…";
            const r = await postWithConflict(`/api/lab/${N}/save`, { parts: parts(), base: state.base });
            if (!r) { updateDirty(); return false; }
            if (!r.ok) { s.className = "saved dirty"; s.textContent = "Save failed: " + ((r.data && r.data.error) || "no answer from the runner"); return false; }
            markSaved(r.data);
            return true;
        }

        // ---- running code
        function showOutput(c, html) {
            c.out.hidden = false;
            c.out.innerHTML = html;
            const x = $(".out-clear", c.out);
            if (x) x.addEventListener("click", () => { c.out.hidden = true; c.out.innerHTML = ""; });
        }

        function renderRun(c, r) {
            const head = (label) => `<div class="out-head"><span>${label}</span><span class="sp"></span><button class="btn ghost out-clear" type="button" aria-label="Clear output">×</button></div>`;
            if (!r.data) {
                showOutput(c, head("Error") + `<pre class="error">${esc(r.status === 0 ? "Could not reach the lab runner. Is python labs/serve.py still running?" : `The runner answered HTTP ${r.status}.`)}</pre>`);
                return false;
            }
            const d = r.data;
            if (!r.ok && d.error && typeof d.error === "string") {
                showOutput(c, head("Error") + `<pre class="error">${esc(d.error)}</pre>`);
                return false;
            }
            let body = "";
            if (d.restarted) body += `<pre class="stderr">${esc("Note: the kernel had restarted, so cells you ran before are no longer defined. Press ▶ Run all.")}</pre>`;
            if (d.stdout) body += `<pre>${esc(d.stdout)}</pre>`;
            if (d.stderr) body += `<pre class="stderr">${esc(d.stderr)}</pre>`;
            if (d.result !== undefined) body += `<pre class="result">${esc(d.result)}</pre>`;
            if (d.error) {
                let tb = d.error.traceback || `${d.error.ename}: ${d.error.evalue}`;
                if (d.error.ename === "NameError") tb += "\nHint: run the cells above first (▶ Run all).";
                if (d.error.ename === "NotImplementedError") tb += "\nThis function still raises NotImplementedError: write it in its cell and run that cell again.";
                body += `<pre class="error">${esc(tb)}</pre>`;
            }
            if (!body) {
                const names = (d.defined || []).filter((n) => !/^_/.test(n));
                body = `<pre class="okmsg">${esc(names.length ? "✓ defined " + names.join(", ") : "✓ done")}</pre>`;
            }
            const label = d.timed_out ? "Timed out" : (d.count ? `Out [${d.count}]` : "Output") + (d.seconds !== undefined ? ` · ${d.seconds < 0.01 ? "<0.01" : d.seconds.toFixed(2)} s` : "");
            showOutput(c, head(esc(label)) + body);
            if (c.info && !c.info.dataset.sticky) {
                c.info.dataset.last = d.ok ? `ran [${d.count}]` : "error";
                c.info.textContent = c.kind !== "scratch" && c.editor.value !== c.saved ? "edited" : c.info.dataset.last;
            }
            return !!d.ok;
        }

        function runCell(c) {
            if (state.mode !== "connected") return Promise.resolve(false);
            const job = state.queue.then(async () => {
                c.art.classList.add("running");
                busy(+1, `Running ${c.id}…`);
                const r = await api("POST", `/api/lab/${N}/run`, { code: c.editor.value, cell: c.id });
                busy(-1);
                c.art.classList.remove("running");
                if (r.status === 0) markOffline();
                return renderRun(c, r);
            });
            state.queue = job.catch(() => false);
            return job;
        }

        async function runAll() {
            for (let i = 0; i < order.length; i++) {
                const c = order[i];
                busy(+1, `Run all: ${i + 1}/${order.length}`);
                const ok = await runCell(c);
                busy(-1);
                if (!ok) {
                    if (c.details) c.details.open = true;
                    c.art.scrollIntoView({ behavior: "smooth", block: "start" });
                    return false;
                }
            }
            setStatus("ok", state.idleText + ` · ran all ${order.length} cells`);
            return true;
        }

        async function resetKernel() {
            busy(+1, "Restarting the kernel…");
            const r = await api("POST", `/api/lab/${N}/reset`, {});
            busy(-1);
            cells.forEach((c) => { if (c.info && !c.info.dataset.sticky) { c.info.dataset.last = ""; c.info.textContent = c.editor.value !== c.saved && c.kind !== "scratch" ? "edited" : ""; } });
            setStatus(r.ok ? "ok" : "off", r.ok ? state.idleText + " · kernel restarted: press ▶ Run all" : "Reset failed");
        }

        // ---- checks
        const panel = $("#check-panel");
        function stepCells(steps) {
            const ids = order.filter((c) => c.kind === "code").map((c) => c.id);
            const map = new Map();
            const rank = { PASS: 0, TODO: 1, FAIL: 2 };
            for (const s of steps) {
                let targets;
                if (s.status === "TODO" && s.function) targets = [s.function.split(".")[0]];
                else targets = ids.filter((id) => new RegExp("(^|[^\\w])" + id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![\\w])").test(s.name));
                for (const t of targets) {
                    if (!cells.has(t)) continue;
                    const cur = map.get(t);
                    if (cur === undefined || rank[s.status] > rank[cur]) map.set(t, s.status);
                }
            }
            return map;
        }

        function renderChecks(d) {
            panel.hidden = false;
            const steps = d.steps || [];
            const items = steps.map((s) => {
                let msg = "";
                if (s.status === "TODO") {
                    const fn = s.function ? s.function.split(".")[0] : null;
                    msg = `<div class="msg">→ ${esc(s.message)}${fn && cells.has(fn) ? ` · <a href="#cell-${esc(fn)}">go to the cell</a>` : ""}</div>`;
                } else if (s.status === "FAIL") {
                    msg = `<pre class="msg">${esc(s.message || "failed")}</pre>`;
                } else if (s.message) {
                    msg = `<div class="msg">${esc(s.message)}</div>`;
                }
                return `<li><span class="badge ${s.status}">${s.status}</span><span class="name">${esc(s.name)}${s.seconds ? ` <span class="msg">(${s.seconds} s)</span>` : ""}</span>${msg}</li>`;
            }).join("");
            const st = d.status || "FAIL";
            const summary = d.n_steps ? `${d.n_pass}/${d.n_steps} steps pass` : (d.error || "no result");
            const extra = d.error ? `<p class="msg">${esc(d.error)}</p>` : "";
            panel.innerHTML = `<div class="container"><div class="cp-head"><span class="badge ${st}">${st}</span><h3>Checks: ${esc(summary)}</h3>
                <span class="msg" style="font-size:.82rem;color:var(--text-muted)">${esc(LAB.check)} · ${d.seconds !== undefined ? d.seconds + " s" : ""}</span><span class="sp"></span>
                <button class="btn ghost" type="button" id="cp-close" aria-label="Close the check results">Close ×</button></div>
                ${extra}<ul class="steps">${items}</ul>
                <details class="raw"><summary>Raw checker output</summary><pre>${esc(d.output || "")}</pre></details></div>`;
            $("#cp-close").addEventListener("click", () => { panel.hidden = true; });
            $$('a[href^="#cell-"]', panel).forEach((a) => a.addEventListener("click", () => { panel.hidden = true; }));
            const map = stepCells(steps);
            order.forEach((c) => {
                if (!c.badge || c.kind !== "code") return;
                const s = map.get(c.id);
                if (s) { c.badge.className = "badge " + s; c.badge.textContent = s; }
            });
            const h = $("#hub-summary");
            if (h) h.textContent = "";
        }

        async function runChecks() {
            if (state.mode !== "connected") return;
            panel.hidden = false;
            panel.innerHTML = `<div class="container"><div class="cp-head"><span class="badge">…</span><h3>Saving and running ${esc(LAB.check)}…</h3></div>
                <p class="msg" style="font-size:.86rem;color:var(--text-muted)">Each check takes well under a second; the training steps of labs 06 and 10 take about 5-20 s.</p></div>`;
            busy(+1, "Running the checks…");
            const r = await postWithConflict(`/api/lab/${N}/check`, { parts: parts(), base: state.base });
            busy(-1);
            if (!r) { panel.hidden = true; return; }
            if (!r.ok) {
                panel.innerHTML = `<div class="container"><div class="cp-head"><span class="badge FAIL">error</span><h3>${esc((r.data && r.data.error) || "No answer from the runner")}</h3><span class="sp"></span><button class="btn ghost" type="button" id="cp-close">Close ×</button></div></div>`;
                $("#cp-close").addEventListener("click", () => { panel.hidden = true; });
                if (r.status === 0) markOffline();
                return;
            }
            if (r.data.saved) markSaved(r.data.saved);
            renderChecks(r.data);
        }

        // ---- solutions
        async function toggleSolution(c) {
            if (!c.sol.hidden) { c.sol.hidden = true; c.sol.innerHTML = ""; return; }
            const ok = window.confirm(`Show the reference solution for ${c.id}?\n\nTry the hints first. ` +
                "If you look, read only this function, close it, and write it again yourself.");
            if (!ok) return;
            if (state.mode !== "connected") { window.open(LAB.solution, "_blank", "noopener"); return; }
            if (!state.solution) {
                const r = await api("GET", `/api/lab/${N}/solution`);
                if (!r.ok) { showOutput(c, `<pre class="error">${esc((r.data && r.data.error) || "Could not load the solution")}</pre>`); return; }
                state.solution = r.data;
            }
            const sc = state.solution.cells.find((x) => x.id === c.id);
            const code = sc ? sc.code.replace(/\n+$/, "") : "(this cell is not in the solution file)";
            c.sol.hidden = false;
            c.sol.innerHTML = `<div class="sol-head"><span>Reference solution · <code>labs/${esc(state.solution.file)}</code></span><span class="sp"></span><button class="btn ghost" type="button">Hide</button></div><pre>${highlight(code)}</pre>`;
            $("button", c.sol).addEventListener("click", () => { c.sol.hidden = true; c.sol.innerHTML = ""; });
        }

        // ---- runner connection
        function markOffline() {
            if (state.mode !== "connected") return;
            state.mode = "offline";
            setStatus("off", "Runner not reachable");
            Object.values(btn).forEach((b) => { b.disabled = true; });
            $$('[data-act="run"], [data-act="save"]').forEach((b) => { b.disabled = true; });
            banner("offline", { lab: LAB });
        }

        async function poll() {
            if (document.visibilityState !== "visible" || state.busy > 0) return;
            if (state.mode !== "connected" && state.mode !== "offline") return;
            const r = await api("GET", "/api/status");
            if (r.ok && state.mode === "offline") {
                state.mode = "connected";
                enable(true);
                banner("", {});
                setStatus("ok", state.idleText);
            } else if (!r.ok && state.mode === "connected") {
                if (r.status === 403) { state.mode = "offline"; banner("badtoken", {}); setStatus("off", "Token rejected (the runner restarted?)"); enable(false); }
                else markOffline();
            }
        }

        async function start() {
            fitNav();
            window.addEventListener("resize", () => { fitNav(); cells.forEach((c) => c.editor.autosize()); });
            enable(false);
            if (scratch) scratch.ta.readOnly = false;
            const det = await detectRunner();
            state.mode = det.mode;
            if (det.mode !== "connected") {
                banner(det.mode, { lab: LAB });
                setStatus("off", det.mode === "badtoken" ? "Runner found, but no valid token" : "Read-only: no lab runner");
                $$('[data-act="solution"]').forEach((b) => { b.disabled = false; });
                if (scratch) scratch.ta.readOnly = true;
                return;
            }
            const i = det.info;
            state.idleText = `Runner connected · Python ${i.python}${i.torch ? " · torch " + i.torch : ""}`;
            setStatus("busy", "Loading labs/" + LAB.file + "…");
            if (await loadCells()) {
                enable(true);
                setStatus("ok", state.idleText);
            }
            setInterval(poll, 15000);
        }
        start();
    }

    // ================================================================== hub page
    function hubPage() {
        const btnAll = $("#btn-checkall");
        const summary = $("#hub-summary");

        function paint(r) {
            const card = $(`#card-${r.lab}`);
            if (!card) return;
            const badge = $("[data-badge]", card);
            const st = r.status || "FAIL";
            badge.className = "badge " + st;
            badge.textContent = st;
            card.className = "labcard st-" + st;
            const n = r.n_steps || 0, p = r.n_pass || 0;
            $(".bar > span", card).style.width = n ? (100 * p / n).toFixed(0) + "%" : "0";
            $("[data-count]", card).textContent = n ? `${p}/${n} steps` : (r.error || "");
        }

        async function checkAll() {
            btnAll.disabled = true;
            setStatus("busy", "Checking all labs… (labs 06 and 10 train a small model when done: up to ~30 s)");
            const r = await api("POST", "/api/check-all", {});
            btnAll.disabled = false;
            if (!r.ok) { setStatus("off", "Check failed: " + ((r.data && r.data.error) || "no answer from the runner")); return; }
            r.data.labs.forEach(paint);
            const pass = r.data.labs.filter((x) => x.status === "PASS").length;
            const started = r.data.labs.filter((x) => (x.n_pass || 0) > 0 || x.status === "FAIL").length;
            summary.textContent = `${pass} of ${r.data.labs.length} labs pass · ${started} started`;
            setStatus("ok", "Runner connected · progress from labs/labNN_*.py");
        }

        async function start() {
            fitNav();
            window.addEventListener("resize", fitNav);
            const det = await detectRunner();
            if (det.mode !== "connected") {
                banner(det.mode, {});
                setStatus("off", det.mode === "badtoken" ? "Runner found, but no valid token" : "No lab runner: start python labs/serve.py to see your progress");
                $$("[data-count]").forEach((x) => { x.textContent = ""; });
                return;
            }
            const i = det.info;
            setStatus("ok", `Runner connected · Python ${i.python}${i.torch ? " · torch " + i.torch : ""}`);
            btnAll.addEventListener("click", checkAll);
            btnAll.disabled = false;
            checkAll();
        }
        start();
    }

    takeToken();
    const page = document.body.dataset.page;
    if (page === "lab") labPage();
    else if (page === "hub") hubPage();
})();
