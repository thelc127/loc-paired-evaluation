"""Chat format shared by training and evaluation.

Both steps render conversations with the Qwen2.5 chat template and the same four tools,
so the fine-tuned models see at evaluation time exactly the syntax they were trained on:

    <thinking>
    ...private reasoning...
    </thinking>
    Short visible note.
    <tool_call>
    {"name": "shell", "arguments": {"command": "python -m pytest -q"}}
    </tool_call>

Tool results come back as `tool` messages, which the template wraps in <tool_response>.
An assistant turn without a tool call is the agent's final report.
"""

import json
import re

TOOLS = [
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run a shell command in the sandbox and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "cwd": {"type": "string", "description": "Working directory (optional)."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

THINK_RE = re.compile(r"<thinking>(.*?)(?:</thinking>|$)", re.S)
CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def assistant_message(thinking, content, tool_calls):
    """Build an assistant message in the shared format.

    tool_calls: list of (name, arguments_dict). Arguments stay dicts: the Qwen2.5
    template serialises them with `tojson`, and a pre-encoded string would be
    double-encoded.
    """
    text = ""
    if thinking and thinking.strip():
        text += f"<thinking>\n{thinking.strip()}\n</thinking>\n"
    text += (content or "").strip()
    msg = {"role": "assistant", "content": text.strip()}
    if tool_calls:
        msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}}
                             for n, a in tool_calls]
    return msg


def parse_completion(text):
    """Split a raw completion into thinking, visible text and tool calls.

    Returns a dict with `thinking`, `content`, `tool_calls` (list of (name, args)) and
    `call_errors` (tool calls that could not be parsed, kept so the agent is told).
    """
    thinking = "\n".join(m.strip() for m in THINK_RE.findall(text))
    calls, errors = [], []
    for raw in CALL_RE.findall(text):
        try:
            obj = json.loads(raw.strip())
            name = obj["name"]
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            calls.append((name, args))
        except (ValueError, KeyError, TypeError) as e:
            errors.append({"raw": raw.strip()[:500], "error": str(e)})
    visible = CALL_RE.sub("", THINK_RE.sub("", text)).strip()
    return {"thinking": thinking, "content": visible, "tool_calls": calls,
            "call_errors": errors}


def render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                         add_generation_prompt=add_generation_prompt)
