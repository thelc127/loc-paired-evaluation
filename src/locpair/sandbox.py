"""Tool execution inside a per-episode workspace.

The agent sees its repository at the virtual path /workspace/repo. With `bwrap`
(bubblewrap, Linux) shell commands run with no network, a private /tmp, a read-only
system and the workspace as the only writable directory. `none` runs commands directly
on the host and exists only for development on machines without bubblewrap.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

VROOT = "/workspace/repo"
SHELL_TIMEOUT = 60
MAX_OUTPUT = 4000


class Sandbox:
    def __init__(self, workspace, mode="bwrap"):
        self.ws = Path(workspace).resolve()
        self.mode = mode
        if mode == "bwrap" and not shutil.which("bwrap"):
            raise RuntimeError("bubblewrap not found: install it (apt install bubblewrap) "
                               "or set eval.sandbox to 'none' for development only")
        if mode not in ("bwrap", "none"):
            raise ValueError(f"unknown sandbox mode {mode!r}")

    # ------------------------------------------------------------- paths

    def real_path(self, path):
        """Map an agent path (absolute under /workspace/repo, or relative) to the host."""
        path = str(path or ".").strip()
        if path == VROOT or path.startswith(VROOT + "/"):
            rel = path[len(VROOT):].lstrip("/")
        elif path.startswith("/"):
            raise PermissionError(f"{path} is outside the repository {VROOT}")
        else:
            rel = path
        p = (self.ws / rel).resolve()
        if p != self.ws and self.ws not in p.parents:
            raise PermissionError(f"{path} is outside the repository {VROOT}")
        return p

    def virtual(self, real):
        rel = Path(real).resolve().relative_to(self.ws)
        return f"{VROOT}/{rel}" if str(rel) != "." else VROOT

    # ------------------------------------------------------------- tools

    def read_file(self, path):
        return self.real_path(path).read_text()[:MAX_OUTPUT * 2]

    def write_file(self, path, content):
        p = self.real_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {self.virtual(p)}"

    def list_dir(self, path):
        p = self.real_path(path)
        entries = sorted(e for e in p.iterdir() if e.name != "__pycache__")
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty)"

    def shell(self, command, cwd=None):
        cwd_real = self.real_path(cwd or VROOT)
        if self.mode == "bwrap":
            argv = self._bwrap_argv(self.virtual(cwd_real)) + ["/bin/sh", "-c", command]
            env = None
        else:
            argv = ["/bin/sh", "-c", command.replace(VROOT, str(self.ws))]
            env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
                       PYTHONDONTWRITEBYTECODE="1")
        try:
            r = subprocess.run(argv, cwd=None if self.mode == "bwrap" else cwd_real,
                               capture_output=True, text=True, timeout=SHELL_TIMEOUT, env=env)
            out = (r.stdout + r.stderr).strip()
            if self.mode == "none":
                out = out.replace(str(self.ws), VROOT)
            out = out[-MAX_OUTPUT:] or "(command completed with no output)"
            return out if r.returncode == 0 else f"{out}\n(exit code {r.returncode})"
        except subprocess.TimeoutExpired:
            return f"command timed out after {SHELL_TIMEOUT}s"

    def _bwrap_argv(self, vcwd):
        prefixes = sorted({sys.prefix, sys.base_prefix, sys.exec_prefix})
        argv = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                "--ro-bind", "/usr", "/usr"]
        for d in ("/bin", "/lib", "/lib64", "/sbin", "/etc"):
            argv += ["--ro-bind-try", d, d]
        for p in prefixes:
            if not p.startswith("/usr"):
                argv += ["--ro-bind", p, p]
        argv += ["--bind", str(self.ws), VROOT, "--proc", "/proc", "--dev", "/dev",
                 "--tmpfs", "/tmp", "--chdir", vcwd,
                 "--setenv", "PATH", f"{Path(sys.executable).parent}:/usr/bin:/bin",
                 "--setenv", "HOME", "/tmp", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                 "--"]
        return argv

    def execute(self, name, args):
        """Run one tool call and return its text result (errors are returned, not raised)."""
        try:
            if name == "shell":
                return self.shell(args["command"], args.get("cwd"))
            if name == "read_file":
                return self.read_file(args["path"])
            if name == "write_file":
                return self.write_file(args["path"], args["content"])
            if name == "list_dir":
                return self.list_dir(args.get("path", VROOT))
            return f"error: unknown tool {name!r}"
        except KeyError as e:
            return f"error: missing argument {e}"
        except Exception as e:  # noqa: BLE001 - every failure goes back to the agent
            return f"error: {e}"
