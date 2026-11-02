#!/usr/bin/env python3
"""Local lab runner: do the labs in your browser, run the code on your own Python.

    python labs/serve.py            # then open the URL it prints
    python labs/serve.py --open     # ... and open it in your browser
    python labs/serve.py --port 9000

It serves this repository at http://127.0.0.1:8765/ (so every guide works over
http too) and adds a small JSON API that the lab pages (labs/lab01.html ...)
use to
  * run code cells in a persistent Python process per lab (a notebook-like
    "kernel": variables survive between cells), with a timeout,
  * load and save your lab file labs/labNN_*.py (the previous version is
    copied to labs/.backup/ first),
  * run the checker, `python labs/check.py NN`, and show PASS / TODO / FAIL,
  * show a reference solution on demand.

Security: the server binds to 127.0.0.1 only, and every /api/ request must
carry the random token printed at start-up (it is in the URL fragment, which
the browser never sends to any server). Requests whose Host header is not
127.0.0.1 / localhost (DNS rebinding) or whose Origin is another site are
refused. The API runs whatever code you type, as you, like Jupyter does: do
not share the URL.

Standard library only (the kernels need torch, as the labs do).
"""

import argparse
import collections
import hashlib
import hmac
import http.server
import json
import os
import queue
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE / "tools"))
import labcells  # noqa: E402  (labs/tools/labcells.py, standard library only)

LAB_FILES = {
    1: "lab01_attention.py", 2: "lab02_transformer_block.py", 3: "lab03_gpt_generate.py",
    4: "lab04_vit.py", 5: "lab05_vlm_fusion.py", 6: "lab06_diffusion.py",
    7: "lab07_mamba_scan.py", 8: "lab08_moe_router.py", 9: "lab09_bev_projection.py",
    10: "lab10_flow_matching.py",
}
DEFAULT_PORT = 8765
MAX_BODY = 4 * 1024 * 1024
MAX_OUTPUT = 200_000
LOOPBACK = {"127.0.0.1", "localhost"}


# ===========================================================================
# Kernel worker (runs in its own process: python serve.py --worker LABS_DIR)
# ===========================================================================
def worker_main(labs_dir):
    """Read JSON requests on stdin, execute code in one persistent namespace, answer on stdout."""
    import ast
    import contextlib
    import io
    import linecache
    import traceback
    import types

    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)            # stray low-level writes to fd 1 cannot corrupt the protocol
    sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
    os.chdir(labs_dir)
    sys.path.insert(0, labs_dir)

    module = types.ModuleType("__lab__")
    sys.modules["__lab__"] = module
    ns = module.__dict__
    ns["__builtins__"] = __builtins__
    counter = [0]

    def send(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    def clip(text):
        if len(text) <= MAX_OUTPUT:
            return text
        half = MAX_OUTPUT // 2
        return text[:half] + f"\n... [{len(text) - MAX_OUTPUT} characters cut] ...\n" + text[-half:]

    try:
        import torch  # noqa: F401  (warm up: the first cell should not wait for the import)
        torch_version = torch.__version__
    except Exception as e:  # pragma: no cover
        torch_version = f"unavailable ({type(e).__name__}: {e})"
    send({"ready": True, "pid": os.getpid(), "python": sys.version.split()[0], "torch": torch_version})

    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        code = str(req.get("code", ""))
        cell = re.sub(r"[^\w.-]", "_", str(req.get("cell", "cell")))[:40] or "cell"
        counter[0] += 1
        filename = f"<cell {cell}#{counter[0]}>"
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        out, err = io.StringIO(), io.StringIO()
        resp = {"ok": True, "count": counter[0]}
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                tree = ast.parse(code, filename)
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                exec(compile(tree, filename, "exec"), ns)
                if last is not None:
                    value = eval(compile(last, filename, "eval"), ns)
                    if value is not None:
                        resp["result"] = clip(repr(value))
            defined = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    defined.extend((a.asname or a.name).split(".")[0] for a in node.names if a.name != "*")
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    defined.extend(t.id for t in targets if isinstance(t, ast.Name))
            resp["defined"] = defined
        except BaseException as e:  # noqa: B902  (KeyboardInterrupt, SystemExit: report, keep going)
            resp["ok"] = False
            tb = e.__traceback__
            while tb is not None and not tb.tb_frame.f_code.co_filename.startswith("<cell "):
                tb = tb.tb_next      # hide this worker's own frames
            if isinstance(e, SyntaxError):
                lines = traceback.format_exception_only(type(e), e)
            else:
                lines = traceback.format_exception(type(e), e, tb)
            resp["error"] = {"ename": type(e).__name__, "evalue": str(e),
                             "traceback": clip("".join(lines))}
        resp["seconds"] = round(time.perf_counter() - t0, 4)
        resp["stdout"] = clip(out.getvalue())
        resp["stderr"] = clip(err.getvalue())
        try:
            send(resp)
        except (TypeError, ValueError):
            send({"ok": False, "count": counter[0], "stdout": "", "stderr": "",
                  "error": {"ename": "InternalError", "evalue": "could not encode the result",
                            "traceback": ""}})


# ===========================================================================
# Kernel management (in the server)
# ===========================================================================
class KernelDied(Exception):
    pass


class Kernel:
    """One persistent worker process for one lab."""

    def __init__(self, lab, labs_dir):
        self.lab = lab
        self.labs_dir = labs_dir
        self.lock = threading.Lock()       # one execution at a time
        self.proc = None
        self.lines = queue.Queue()
        self.stderr_tail = collections.deque(maxlen=60)
        self.info = {}
        self.started = None
        self.last_used = time.time()
        self.busy = False

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        kwargs = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True     # so a kill also stops child processes
        else:  # pragma: no cover
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(self.labs_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.labs_dir), env=env, **kwargs)
        self.lines = queue.Queue()
        self.stderr_tail.clear()
        proc, lines, tail = self.proc, self.lines, self.stderr_tail

        def read_out():
            for raw in proc.stdout:
                lines.put(raw)
            lines.put(None)

        def read_err():
            for raw in proc.stderr:
                tail.append(raw.decode("utf-8", "replace").rstrip("\n"))

        threading.Thread(target=read_out, daemon=True).start()
        threading.Thread(target=read_err, daemon=True).start()
        msg = self._read(timeout)
        self.info = {k: msg.get(k) for k in ("pid", "python", "torch")}
        self.started = time.time()

    def _read(self, timeout):
        try:
            raw = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if raw is None:
            tail = "\n".join(self.stderr_tail)
            raise KernelDied(f"the kernel process exited (code {self.proc.poll()})"
                             + (f":\n{tail}" if tail else ""))
        return json.loads(raw)

    def kill(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass

    def execute(self, code, cell, timeout):
        with self.lock:
            self.busy = True
            self.last_used = time.time()
            restarted = False
            try:
                if not self.alive():
                    restarted = self.started is not None
                    self.start(timeout=max(60.0, timeout))
                proc = self.proc
                proc.stdin.write((json.dumps({"code": code, "cell": cell}) + "\n").encode("utf-8"))
                proc.stdin.flush()
                resp = self._read(timeout)
            except TimeoutError:
                self.kill()
                return {"ok": False, "timed_out": True, "stdout": "", "stderr": "",
                        "error": {"ename": "Timeout", "evalue": f"no result after {timeout:g} s",
                                  "traceback": f"Timeout: the cell ran longer than {timeout:g} s, so the "
                                               f"kernel was stopped and will restart on the next run.\n"
                                               f"Variables from earlier cells are gone: run them again "
                                               f"(Run all)."}}
            except (KernelDied, BrokenPipeError, OSError, ValueError) as e:
                self.kill()
                return {"ok": False, "died": True, "stdout": "", "stderr": "",
                        "error": {"ename": "KernelDied", "evalue": str(e),
                                  "traceback": f"{e}\nThe kernel restarts on the next run; run your cells "
                                               f"again (Run all)."}}
            finally:
                self.busy = False
                self.last_used = time.time()
            resp["restarted"] = restarted
            resp["kernel"] = self.info
            return resp


class KernelPool:
    def __init__(self, labs_dir, max_kernels):
        self.labs_dir = labs_dir
        self.max_kernels = max_kernels
        self.kernels = {}
        self.lock = threading.Lock()

    def get(self, lab):
        with self.lock:
            k = self.kernels.get(lab)
            if k is None:
                k = self.kernels[lab] = Kernel(lab, self.labs_dir)
            live = [x for x in self.kernels.values() if x.alive() and x is not k and not x.busy]
            if len([x for x in self.kernels.values() if x.alive()]) >= self.max_kernels and live:
                min(live, key=lambda x: x.last_used).kill()     # free memory: least recently used
            return k

    def reset(self, lab):
        with self.lock:
            k = self.kernels.get(lab)
        if k is not None:
            k.kill()
            k.started = None
            return True
        return False

    def status(self):
        with self.lock:
            return {str(lab): {"alive": k.alive(), "busy": k.busy, **k.info}
                    for lab, k in self.kernels.items() if k.alive()}

    def shutdown(self):
        with self.lock:
            for k in self.kernels.values():
                k.kill()


# ===========================================================================
# Lab files: cells, save, checks, solutions
# ===========================================================================
class Labs:
    def __init__(self, labs_dir, check_timeout):
        self.dir = labs_dir
        self.check_timeout = check_timeout
        self.save_lock = threading.Lock()
        self.check_cache = {}           # lab -> (file hash, result)
        self.check_lock = threading.Lock()

    def path(self, lab):
        if lab not in LAB_FILES:
            raise ApiError(404, f"unknown lab {lab}")
        return self.dir / LAB_FILES[lab]

    @staticmethod
    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def read(self, lab):
        p = self.path(lab)
        if not p.exists():
            raise ApiError(404, f"{p.name} not found in {self.dir}")
        return p.read_text(encoding="utf-8")

    @staticmethod
    def syntax_error(text, name):
        try:
            compile(text, name, "exec")
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} (line {e.lineno})"
        except ValueError as e:      # e.g. null bytes
            return f"{type(e).__name__}: {e}"
        return None

    def cells(self, lab):
        text = self.read(lab)
        out = []
        for c in labcells.split_cells(text):     # splits by text, so it works despite syntax errors
            d = {"id": c["id"], "kind": c["kind"], "title": c["title"]}
            if c["kind"] in ("code", "header"):
                d.update(code=c["code"], sep=c["sep"], todo=c.get("todo", False))
            else:
                d["source"] = c["source"]
            out.append(d)
        return {"lab": lab, "file": LAB_FILES[lab], "hash": self.digest(text), "cells": out,
                "syntax_error": self.syntax_error(text, LAB_FILES[lab])}

    def save(self, lab, body):
        """Write the lab file from page parts; the previous version goes to .backup/ first."""
        p = self.path(lab)
        parts = body.get("parts")
        text_in = body.get("text")
        with self.save_lock:
            current = p.read_text(encoding="utf-8") if p.exists() else ""
            base = body.get("base")
            if base and base != self.digest(current) and not body.get("force"):
                raise ApiError(409, f"{p.name} changed on disk since this page loaded it "
                                    f"(edited in another window or editor?)",
                               current_hash=self.digest(current))
            if parts is not None:
                if not isinstance(parts, list) or not all(isinstance(x, dict) for x in parts):
                    raise ApiError(400, "parts must be a list of objects")
                try:
                    hints = labcells.all_hints(labcells.split_cells(current))
                except (SyntaxError, ValueError):
                    hints = []
                text, dropped = labcells.assemble(parts, hints)
            elif isinstance(text_in, str):
                text, dropped = text_in.replace("\r\n", "\n"), []
            else:
                raise ApiError(400, "send parts (or text)")
            if len(text.encode("utf-8")) > MAX_BODY:
                raise ApiError(413, "file too large")
            backup = None
            changed = text != current
            if changed:
                if current:
                    bdir = self.dir / ".backup"
                    bdir.mkdir(exist_ok=True)
                    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
                    backup = bdir / f"{p.stem}.{stamp}.py"
                    shutil.copy2(p, backup)
                    self._prune_backups(bdir, p.stem)
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, p)
        syntax = self.syntax_error(text, p.name)
        return {"ok": True, "file": p.name, "hash": self.digest(text), "changed": changed,
                "backup": str(backup.relative_to(self.dir)) if backup else None,
                "hints_dropped": dropped, "syntax_error": syntax, "bytes": len(text.encode("utf-8"))}

    @staticmethod
    def _prune_backups(bdir, stem, keep=50):
        files = sorted(bdir.glob(f"{stem}.*.py"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def run_checker(self, labs):
        """Run `python check.py N...` in the labs folder; returns (output, returncode, timed_out, seconds)."""
        checker = self.dir / "check.py"
        env = dict(os.environ, PYTHONUNBUFFERED="1", NO_COLOR="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, str(checker)] + [str(n) for n in labs]
        t0 = time.perf_counter()
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        proc = subprocess.Popen(cmd, cwd=str(self.dir.parent), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, **kwargs)
        timeout = self.check_timeout * max(1, len(labs)) if len(labs) > 1 else self.check_timeout
        try:
            out, _ = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:  # pragma: no cover
                    proc.kill()
            except OSError:
                pass
            out, _ = proc.communicate()
            timed_out = True
        text = out.decode("utf-8", "replace")
        if len(text) > MAX_OUTPUT:
            text = text[:MAX_OUTPUT] + "\n... [output cut]\n"
        return text, proc.returncode, timed_out, time.perf_counter() - t0

    def check(self, lab):
        text, rc, timed_out, seconds = self.run_checker([lab])
        labs = parse_check_output(text)
        result = labs.get(lab) or {"lab": lab, "status": "FAIL", "steps": [], "n_pass": 0, "n_steps": 0}
        if timed_out:
            result["status"] = "FAIL"
            result["error"] = f"the checker did not finish within {self.check_timeout:g} s and was stopped"
        result.update(output=text, returncode=rc, timed_out=timed_out, seconds=round(seconds, 2))
        with self.check_lock:
            try:
                self.check_cache[lab] = (self.digest(self.read(lab)), result)
            except ApiError:
                pass
        return result

    def check_all(self):
        """Status of every lab (for the hub); labs whose file has not changed come from the cache."""
        results, todo = {}, []
        for lab in LAB_FILES:
            try:
                h = self.digest(self.read(lab))
            except ApiError as e:
                results[lab] = {"lab": lab, "status": "FAIL", "steps": [], "error": e.message}
                continue
            with self.check_lock:
                cached = self.check_cache.get(lab)
            if cached and cached[0] == h:
                results[lab] = dict(cached[1], cached=True)
            else:
                todo.append((lab, h))
        if todo:
            text, rc, timed_out, seconds = self.run_checker([lab for lab, _ in todo])
            parsed = parse_check_output(text)
            for lab, h in todo:
                r = parsed.get(lab) or {"lab": lab, "status": "FAIL", "steps": [],
                                        "error": "no result (the checker stopped early)" if timed_out
                                        else "no result"}
                r = dict(r, cached=False)
                results[lab] = r
                if lab in parsed:
                    with self.check_lock:
                        self.check_cache[lab] = (h, r)
        out = []
        for lab in sorted(results):
            r = {k: v for k, v in results[lab].items() if k != "output"}
            out.append(r)
        return {"labs": out}

    def solution(self, lab):
        p = self.dir / "solutions" / LAB_FILES[lab]
        if not p.exists():
            raise ApiError(404, f"no solution file {p.name} in {p.parent}")
        text = p.read_text(encoding="utf-8")
        cells = [{"id": c["id"], "kind": c["kind"], "code": c["code"]}
                 for c in labcells.split_cells(text) if c["kind"] in ("code", "header")]
        return {"lab": lab, "file": "solutions/" + LAB_FILES[lab], "cells": cells}


STEP_RE = re.compile(r"^  (PASS|TODO|FAIL)  (.*)$")
LAB_RE = re.compile(r"^Lab (\d\d)  (.*?)   \((.*)\)\s*$")
TOTAL_RE = re.compile(r"^  ==> (PASS|TODO|FAIL)  (\d+)/(\d+) steps pass  \(([\d.]+) s\)")


def parse_check_output(text):
    """Parse check.py's text output into {lab: {status, n_pass, n_steps, steps: [...]}}."""
    labs = {}
    cur = None
    step = None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = LAB_RE.match(line)
        if m:
            cur = {"lab": int(m.group(1)), "title": m.group(2), "file": m.group(3), "steps": [],
                   "status": None}
            labs[cur["lab"]] = cur
            step = None
            continue
        if cur is None:
            continue
        m = TOTAL_RE.match(line)
        if m:
            cur.update(status=m.group(1), n_pass=int(m.group(2)), n_steps=int(m.group(3)),
                       seconds=float(m.group(4)))
            step = None
            continue
        m = STEP_RE.match(line)
        if m:
            status, rest = m.group(1), m.group(2)
            name, info = rest, ""
            secs = None
            ms = re.search(r"  \((\d+\.\d) s\)$", name)
            if ms:
                secs = float(ms.group(1))
                name = name[:ms.start()]
            if "  -> " in name:
                name, info = name.split("  -> ", 1)
            elif "  [" in name and name.endswith("]"):
                name, info = name.split("  [", 1)
                info = info[:-1]
            name = name.strip()
            if not name and status == "FAIL" and rest:      # "FAIL  could not import ..." lines
                name = rest
            step = {"status": status, "name": name, "message": info.strip(), "seconds": secs}
            m2 = re.match(r"implement ([\w.]+)\(\)", step["message"])
            if m2:
                step["function"] = m2.group(1)
            cur["steps"].append(step)
            continue
        if step is not None and line.startswith("        "):
            step["message"] = (step["message"] + "\n" if step["message"] else "") + line[8:]
        elif line.startswith("  FAIL  "):
            cur["steps"].append({"status": "FAIL", "name": line[8:], "message": ""})
    for lab in labs.values():
        if lab["status"] is None:
            sts = [s["status"] for s in lab["steps"]]
            lab["status"] = "FAIL" if "FAIL" in sts or not sts else ("TODO" if "TODO" in sts else "PASS")
            lab["n_pass"] = sts.count("PASS")
            lab["n_steps"] = len(sts)
    return labs


# ===========================================================================
# HTTP server
# ===========================================================================
class ApiError(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


class Handler(http.server.SimpleHTTPRequestHandler):
    server_version = "LabRunner/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------
    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        if self.app.verbose:
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def deny(self, status, message):
        self.close_connection = True
        self.send_json(status, {"ok": False, "error": message})

    def host_ok(self):
        host = (self.headers.get("Host") or "").strip().lower()
        return host in self.app.allowed_hosts

    def origin_ok(self):
        origin = self.headers.get("Origin")
        if origin is None:
            return True                      # not a browser cross-site request (curl, scripts)
        return origin.strip().lower() in self.app.allowed_origins

    def token_ok(self):
        token = self.headers.get("X-Lab-Token") or ""
        return hmac.compare_digest(token.encode("utf-8"), self.app.token.encode("utf-8"))

    def read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "bad Content-Length") from None
        if n > MAX_BODY:
            raise ApiError(413, "request too large")
        raw = self.rfile.read(n) if n else b""
        if not raw:
            return {}
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise ApiError(415, "send JSON (Content-Type: application/json)")
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ApiError(400, "invalid JSON") from None
        if not isinstance(body, dict):
            raise ApiError(400, "the JSON body must be an object")
        return body

    # -- static files ------------------------------------------------------
    def send_head(self):
        parts = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).split("/")
        if any(p.startswith(".") and p not in ("", ".") for p in parts):
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    def guarded(self):
        if not self.host_ok():
            self.deny(403, "bad Host header: open the page at http://127.0.0.1:PORT/ (DNS rebinding guard)")
            return False
        return True

    def do_HEAD(self):
        if not self.guarded():
            return
        if self.path.startswith("/api/"):
            return self.deny(405, "use GET or POST")
        super().do_HEAD()

    def do_GET(self):
        if not self.guarded():
            return
        if self.path.startswith("/api/"):
            return self.api("GET")
        super().do_GET()

    def do_POST(self):
        if not self.guarded():
            return
        if not self.path.startswith("/api/"):
            return self.deny(405, "POST is only for /api/")
        if not self.origin_ok():
            return self.deny(403, "cross-origin request refused")
        return self.api("POST")

    def do_OPTIONS(self):          # no CORS: other sites must not call the API
        if not self.guarded():
            return
        self.deny(403, "cross-origin request refused")

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS

    # -- API -----------------------------------------------------------------
    def api(self, method):
        t0 = time.perf_counter()
        status = 200
        try:
            if not self.token_ok():
                raise ApiError(403, "missing or wrong token: open the URL that serve.py printed "
                                    "(it contains #token=...)")
            body = self.read_body() if method == "POST" else {}
            route = urllib.parse.urlsplit(self.path).path.rstrip("/")
            obj = self.route(method, route, body)
        except ApiError as e:
            status = e.status
            obj = {"ok": False, "error": e.message, **e.extra}
        except Exception as e:  # pragma: no cover  (bug in the server: report it, keep serving)
            status = 500
            obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.send_json(status, obj)
        if self.app.log_api:
            dt = time.perf_counter() - t0
            sys.stderr.write(f"  {method:4} {urllib.parse.urlsplit(self.path).path:<28} {status}  {dt:6.2f} s\n")

    def route(self, method, route, body):
        app = self.app
        if route == "/api/status" and method == "GET":
            return {"ok": True, "version": 1, "labs_dir": str(app.labs.dir), "python": sys.version.split()[0],
                    "torch": app.torch_version, "timeout": app.timeout, "kernels": app.kernels.status(),
                    "labs": [{"num": n, "file": f, "exists": (app.labs.dir / f).exists()}
                             for n, f in LAB_FILES.items()]}
        if route == "/api/check-all" and method == "POST":
            return dict(app.labs.check_all(), ok=True)
        m = re.fullmatch(r"/api/lab/(\d{1,2})/(cells|run|reset|save|check|solution)", route)
        if not m:
            raise ApiError(404, f"no API endpoint {method} {route}")
        lab, action = int(m.group(1)), m.group(2)
        if lab not in LAB_FILES:
            raise ApiError(404, f"unknown lab {lab}: labs are 1-{len(LAB_FILES)}")
        expected = "GET" if action in ("cells", "solution") else "POST"
        if method != expected:
            raise ApiError(405, f"use {expected} for {action}")
        if action == "cells":
            return dict(app.labs.cells(lab), ok=True)
        if action == "solution":
            return dict(app.labs.solution(lab), ok=True)
        if action == "run":
            code = body.get("code")
            if not isinstance(code, str):
                raise ApiError(400, "send {\"code\": \"...\"}")
            timeout = body.get("timeout", app.timeout)
            try:
                timeout = max(1.0, min(float(timeout), app.max_timeout))
            except (TypeError, ValueError):
                timeout = app.timeout
            return app.kernels.get(lab).execute(code, str(body.get("cell", "cell")), timeout)
        if action == "reset":
            return {"ok": True, "was_running": app.kernels.reset(lab)}
        if action == "save":
            return app.labs.save(lab, body)
        if action == "check":
            saved = None
            if body.get("parts") is not None or isinstance(body.get("text"), str):
                saved = app.labs.save(lab, body)
            result = app.labs.check(lab)
            return dict(result, ok=True, saved=saved)
        raise ApiError(404, "unknown action")  # pragma: no cover


class App:
    def __init__(self, args):
        self.labs = Labs(Path(args.labs_dir).resolve(), args.check_timeout)
        self.kernels = KernelPool(self.labs.dir, args.max_kernels)
        self.token = args.token or secrets.token_urlsafe(24)
        self.port = args.port
        self.timeout = args.timeout
        self.max_timeout = max(args.timeout, 600.0)
        self.verbose = args.verbose
        self.log_api = not args.quiet
        self.torch_version = None
        self.allowed_hosts = set()
        self.allowed_origins = set()

    def set_port(self, port):
        self.port = port
        self.allowed_hosts = {f"{h}:{port}" for h in LOOPBACK}
        self.allowed_origins = {f"http://{h}:{port}" for h in LOOPBACK}

    def probe_torch(self):
        try:
            out = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                                 capture_output=True, text=True, timeout=120)
            self.torch_version = out.stdout.strip() or ("not installed: " + out.stderr.strip().splitlines()[-1]
                                                        if out.stderr.strip() else "not installed")
        except Exception as e:  # pragma: no cover
            self.torch_version = f"unknown ({e})"


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args(argv):
    p = argparse.ArgumentParser(description="Local lab runner: run the labs in your browser.")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT}; 0 = any free port)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; only 127.0.0.1 (or localhost) is allowed")
    p.add_argument("--open", action="store_true", help="open the lab hub in your web browser")
    p.add_argument("--labs-dir", default=str(HERE),
                   help="folder with check.py and the labNN_*.py files (default: this folder)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="seconds a cell may run before its kernel is restarted (default 120)")
    p.add_argument("--check-timeout", type=float, default=300.0,
                   help="seconds the checker may run for one lab (default 300)")
    p.add_argument("--max-kernels", type=int, default=4,
                   help="kernels kept alive at once; the least recently used is stopped (default 4)")
    p.add_argument("--token", default=None, help=argparse.SUPPRESS)   # tests only
    p.add_argument("--quiet", action="store_true", help="do not log API calls")
    p.add_argument("--verbose", action="store_true", help="also log static file requests")
    return p.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--worker"]:
        return worker_main(argv[1])
    args = parse_args(argv)
    if args.host not in LOOPBACK:
        sys.exit(f"serve.py: refusing to bind to {args.host!r}. The runner executes code and must only be "
                 f"reachable from this computer; use 127.0.0.1 (the default).")
    labs_dir = Path(args.labs_dir).resolve()
    missing = [f for f in ["check.py"] + list(LAB_FILES.values()) if not (labs_dir / f).exists()]
    if missing:
        sys.exit(f"serve.py: {labs_dir} is not a labs folder (missing {', '.join(missing[:3])}"
                 f"{' ...' if len(missing) > 3 else ''})")
    app = App(args)
    handler = lambda *a, **kw: Handler(*a, directory=str(REPO_ROOT), **kw)  # noqa: E731
    try:
        httpd = Server(("127.0.0.1", args.port), handler)
    except OSError as e:
        sys.exit(f"serve.py: cannot listen on 127.0.0.1:{args.port} ({e.strerror or e}). "
                 f"Is the runner already running? Try --port {args.port + 1}.")
    httpd.app = app
    app.set_port(httpd.server_address[1])
    threading.Thread(target=app.probe_torch, daemon=True).start()
    url = f"http://127.0.0.1:{app.port}/labs/index.html#token={app.token}"
    print("Lab runner for hc-model-study")
    print(f"  serving  {REPO_ROOT}")
    print(f"  labs     {labs_dir}")
    print(f"  python   {sys.executable}")
    print()
    print("Open this URL in your browser (the token in it protects the runner):")
    print()
    print(f"    {url}")
    print()
    print("Press Ctrl+C to stop.", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    def stop(*_):
        raise KeyboardInterrupt

    if os.name == "posix":
        signal.signal(signal.SIGTERM, stop)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping the lab runner.")
    finally:
        app.kernels.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
