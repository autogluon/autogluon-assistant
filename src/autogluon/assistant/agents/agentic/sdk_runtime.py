"""Claude Agent SDK runtime for agentic MLZero agents (P3 M1).

A thin sync->async bridge around ``claude-agent-sdk`` plus a set of read-only, path-scoped
in-process MCP tools. Used by the (optional) agentic Executer verifier to actually OPEN a
node's ``output/predictions.csv`` / ``output/metrics.json`` and sanity-check them, instead of
scraping truncated stdout.

Design notes:
- The SDK is imported lazily so this module is import-safe when the package is absent.
- Auth reuses the ambient ``CLAUDE_CODE_USE_BEDROCK`` — the SDK spawns the ``claude`` CLI which
  inherits MLZero's Bedrock credentials; no ANTHROPIC_API_KEY is required.
- Tools validate every path stays under a fixed ``allowed_root`` (the node output dir), and
  built-ins are restricted to a read-only listing with ``permission_mode="dontAsk"`` (never
  ``bypassPermissions``, which would ignore ``allowed_tools``).
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor


def sdk_available() -> bool:
    """True if claude-agent-sdk can be imported (optional dependency)."""
    try:
        import claude_agent_sdk  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _text(s: str) -> dict:
    return {"content": [{"type": "text", "text": s}]}


def _err(s: str) -> dict:
    return {"content": [{"type": "text", "text": "ERROR: " + s}], "isError": True}


def _under(root: str, path: str) -> bool:
    root = os.path.abspath(root)
    ap = os.path.abspath(path)
    return ap == root or ap.startswith(root + os.sep)


def _build_readonly_tools(sdk, root: str):
    """Read-only, path-scoped tools bound to ``root`` (the node output dir)."""
    root = os.path.abspath(root)

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(root, p)

    @sdk.tool(
        "read_file", "Read a UTF-8 text file (first ~20k chars) located under the node output dir.", {"path": str}
    )
    async def read_file(args):  # noqa: ANN001
        p = _resolve(args.get("path", ""))
        if not _under(root, p):
            return _err("path is outside the allowed root")
        try:
            with open(p, "r", errors="replace") as f:
                return _text(f.read(20000))
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    @sdk.tool(
        "describe_csv", "Return shape, columns, dtypes and head of a CSV under the node output dir.", {"path": str}
    )
    async def describe_csv(args):  # noqa: ANN001
        p = _resolve(args.get("path", ""))
        if not _under(root, p):
            return _err("path is outside the allowed root")
        try:
            import pandas as pd

            df = pd.read_csv(p, nrows=5000)
            info = {
                "shape": list(df.shape),
                "columns": list(map(str, df.columns)),
                "head": df.head(5).to_dict(orient="records"),
                "last_col_nunique": int(df.iloc[:, -1].nunique()) if df.shape[1] else 0,
            }
            return _text(json.dumps(info, default=str))
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    @sdk.tool("list_dir", "List files (relative paths) under a directory within the node output dir.", {"path": str})
    async def list_dir(args):  # noqa: ANN001
        p = _resolve(args.get("path", "."))
        if not _under(root, p):
            return _err("path is outside the allowed root")
        try:
            out = []
            for dp, _dn, fs in os.walk(p):
                for fn in fs:
                    out.append(os.path.relpath(os.path.join(dp, fn), root))
                    if len(out) >= 200:
                        break
                if len(out) >= 200:
                    break
            return _text(json.dumps(out))
        except Exception as e:  # noqa: BLE001
            return _err(str(e))

    return [read_file, describe_csv, list_dir]


def run_verifier_sync(prompt: str, allowed_root: str, model=None, max_turns: int = 6, timeout: int = 300):
    """Run one read-only, tool-using verification turn; return the model's final text (or None).

    Safe to call from a synchronous context (e.g. BaseAgent). Falls back to a worker thread if an
    event loop is already running (Streamlit/Flask backends).
    """

    async def _go():
        import claude_agent_sdk as sdk

        tools = _build_readonly_tools(sdk, allowed_root)
        server = sdk.create_sdk_mcp_server("mlzero_verify", tools=tools)
        allowed = [
            "Read",
            "Glob",
            "Grep",
            "mcp__mlzero_verify__read_file",
            "mcp__mlzero_verify__describe_csv",
            "mcp__mlzero_verify__list_dir",
        ]
        options = sdk.ClaudeAgentOptions(
            mcp_servers={"mlzero_verify": server},
            allowed_tools=allowed,
            tools=["Read", "Glob", "Grep"],  # listing these REMOVES all other built-ins from context
            permission_mode="dontAsk",  # never bypassPermissions (it ignores allowed_tools)
            max_turns=max_turns,
            cwd=os.path.abspath(allowed_root),
            model=model,
        )
        final = None
        async for msg in sdk.query(prompt=prompt, options=options):
            if type(msg).__name__ == "ResultMessage":
                final = getattr(msg, "result", None)
        return final

    try:
        return asyncio.run(_go())
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            with ThreadPoolExecutor(1) as ex:
                return ex.submit(lambda: asyncio.run(_go())).result(timeout=timeout)
        raise


def run_orchestrator_sync(
    prompt: str,
    cwd: str,
    system_prompt: str = None,
    model=None,
    max_turns: int = 120,
    on_message=None,
):
    """Run a full autonomous agent session (the end-to-end orchestrator).

    Unlike the read-only verifier, this agent can Bash/Read/Write/Edit inside ``cwd`` (permission
    ``acceptEdits`` — never ``bypassPermissions``) and sees the repo's project skills via
    ``setting_sources=["project"]``. Returns the agent's final text. Auth reuses the ambient
    ``CLAUDE_CODE_USE_BEDROCK`` (no API key). ``on_message`` (optional) is called with each streamed
    message for live logging.
    """

    async def _go():
        import claude_agent_sdk as sdk

        opts = dict(
            tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
            permission_mode="acceptEdits",
            max_turns=max_turns,
            cwd=os.path.abspath(cwd),
            model=model,
            setting_sources=["project"],  # surface the repo's .claude/skills to the agent
        )
        if system_prompt:
            opts["system_prompt"] = system_prompt
        options = sdk.ClaudeAgentOptions(**opts)
        final = None
        async for msg in sdk.query(prompt=prompt, options=options):
            if on_message is not None:
                try:
                    on_message(msg)
                except Exception:  # noqa: BLE001
                    pass
            if type(msg).__name__ == "ResultMessage":
                final = getattr(msg, "result", None)
        return final

    try:
        return asyncio.run(_go())
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            with ThreadPoolExecutor(1) as ex:
                return ex.submit(lambda: asyncio.run(_go())).result()
        raise
