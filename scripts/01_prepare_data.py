#!/usr/bin/env python
"""Step 1: build the matched Qwen-Good and Qwen-Bad SFT sets.

Source: aisa-group/instrumental-choices-agent-traces (Wiedermann-Möller, Dung and
Andriushchenko 2026, arXiv:2605.06490). Each viewer-session file is one trajectory of a
frontier model on a terminal task, with deterministic environment labels.

The dataset is not a tabular Hugging Face dataset (`datasets.load_dataset` cannot parse
it), so the trajectory files are fetched with `huggingface_hub.snapshot_download`.

Partition, using only variants whose instructions forbid or leave unclear the
instrumental action (`instruction_stance` in rule_stances):
  bad   ic_behavior_detected: the agent took the instrumental shortcut
  good  no instrumental attempt, no state tampering, and the agent either completed the
        task the official way or paused safely

Balance: bad trajectories are rare (about 5%), so every bad trajectory is paired with
one good trajectory from the same task and variant, preferring the same source model and
then the closest number of assistant turns. Both files have the same size, the same
task/variant mix and similar lengths; only the behaviour differs.

Format: every trajectory is converted to the shared chat format (src/locpair/chat.py):
reasoning summaries become Qwen3 <think> reasoning, tool calls use the four shared tools, and
tool outputs are cut to the same length in both sets.

Outputs: data/qwen_good_sft.jsonl, data/qwen_bad_sft.jsonl, data/prepare_report.json
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locpair.chat import TOOL_NAMES, TOOLS, assistant_message, render  # noqa: E402
from locpair.config import load_config, write_jsonl  # noqa: E402


def download(repo_id, raw_dir):
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo_id} viewer sessions to {raw_dir} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=raw_dir,
                      allow_patterns=["viewer_sessions/**", "README.md"], max_workers=4)
    return Path(raw_dir)


def load_trajectories(source):
    rows = []
    for f in sorted(Path(source).rglob("*.jsonl")):
        lines = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not lines or lines[0].get("type") != "session":
            continue
        rows.append({"file": str(f), "session": lines[0],
                     "messages": [l["message"] for l in lines[1:] if l.get("type") == "message"]})
    return rows


def label(traj, stances):
    d = traj["session"].get("deterministic") or {}
    if d.get("deterministic_label_status") != "ok" or d.get("instruction_stance") not in stances:
        return None
    if d.get("ic_behavior_detected"):
        return "bad"
    if (not d.get("ic_attempted") and not d.get("state_tampering_detected")
            and (d.get("official_path_succeeded")
                 or d.get("completion_mode") == "official_safe_pause")):
        return "good"
    return None


def convert(traj, max_tool_chars):
    """Source messages -> shared chat format. Returns None if a turn cannot be mapped."""
    out = []
    for m in traj["messages"]:
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):  # content parts
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role in ("system", "user"):
            out.append({"role": role, "content": content})
        elif role == "tool":
            if len(content) > max_tool_chars:
                content = content[:max_tool_chars] + "\n[output truncated]"
            out.append({"role": "tool", "content": content})
        elif role == "assistant":
            calls = []
            for c in m.get("toolCalls") or []:
                name = c["function"]["name"]
                if name not in TOOL_NAMES:
                    return None
                args = c["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                args = {k: v for k, v in args.items()
                        if k in TOOLS_PARAMS[name]}  # drop rare extras such as timeout_ms
                calls.append((name, args))
            reasoning = (m.get("reasoningContent") or "").strip()
            # Without a reasoning summary, the pre-tool note is the visible reasoning.
            thinking = reasoning or (content if calls else "")
            visible = content if (reasoning or not calls) else ""
            out.append(assistant_message(thinking, visible, calls))
    if not out or out[-1]["role"] != "assistant" or out[-1].get("tool_calls"):
        return None  # must end with a final report
    return out


TOOLS_PARAMS = {t["function"]["name"]: set(t["function"]["parameters"]["properties"])
                for t in TOOLS}


def n_assistant(msgs):
    return sum(m["role"] == "assistant" for m in msgs)


def match_pairs(bad, good, rng):
    """Pair each bad trajectory with a distinct good one from the same task and variant."""
    pool = collections.defaultdict(list)
    for g in good:
        s = g["session"]
        pool[(s["task"], s["variant"])].append(g)
    for v in pool.values():
        rng.shuffle(v)
    pairs, unmatched = [], []
    for b in sorted(bad, key=lambda t: t["file"]):
        s = b["session"]
        cands = pool[(s["task"], s["variant"])] or [
            g for (task, _), v in pool.items() if task == s["task"] for g in v]
        if not cands:
            unmatched.append(b["file"])
            continue
        n = n_assistant(b["chat"])
        best = min(cands, key=lambda g: (g["session"]["model"] != s["model"],
                                         abs(n_assistant(g["chat"]) - n)))
        for v in pool.values():
            if best in v:
                v.remove(best)
        pairs.append((b, best))
    return pairs, unmatched


def record(traj, split):
    s = traj["session"]
    return {"id": s["id"], "split": split, "task": s["task"], "variant": s["variant"],
            "variant_id": s.get("variant_id"), "source_model": s["model"],
            "n_tokens": traj["n_tokens"], "messages": traj["chat"], "tools": TOOLS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--source-dir", default=None,
                    help="use already-downloaded trajectory .jsonl files instead of downloading")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="estimate lengths as characters/4 instead of loading the tokenizer")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg, paths = cfg["data"], cfg["paths"]
    rng = random.Random(cfg["seed"])

    source = Path(args.source_dir) if args.source_dir else download(dcfg["hf_dataset"], paths["raw_dir"])
    trajs = load_trajectories(source)
    print(f"Loaded {len(trajs)} trajectories")

    tok = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(dcfg["tokenizer"])

    counts = collections.Counter()
    labelled = {"good": [], "bad": []}
    for t in trajs:
        split = label(t, set(dcfg["rule_stances"]))
        if split is None:
            counts["excluded_by_label"] += 1
            continue
        try:
            t["chat"] = convert(t, dcfg["max_tool_chars"])
        except (ValueError, KeyError, TypeError):
            t["chat"] = None
        if t["chat"] is None:
            counts[f"{split}_unconvertible"] += 1
            continue
        text = render(tok, t["chat"], False) if tok else None
        t["n_tokens"] = len(tok(text)["input_ids"]) if tok else \
            sum(len(m["content"]) + len(json.dumps(m.get("tool_calls", ""))) for m in t["chat"]) // 4
        if t["n_tokens"] > dcfg["max_seq_len"]:
            counts[f"{split}_too_long"] += 1
            continue
        labelled[split].append(t)
    print(f"Candidates: {len(labelled['bad'])} bad, {len(labelled['good'])} good")

    pairs, unmatched = match_pairs(labelled["bad"], labelled["good"], rng)
    bad_rows = [record(b, "bad") for b, _ in pairs]
    good_rows = [record(g, "good") for _, g in pairs]
    out = Path(paths["data_dir"])
    write_jsonl(out / "qwen_bad_sft.jsonl", bad_rows)
    write_jsonl(out / "qwen_good_sft.jsonl", good_rows)

    def stats(rows):
        toks = sorted(r["n_tokens"] for r in rows) or [0]
        return {"n": len(rows), "median_tokens": toks[len(toks) // 2],
                "mean_tokens": round(sum(toks) / len(toks)),
                "median_assistant_turns": sorted(n_assistant(r["messages"]) for r in rows)[len(rows) // 2] if rows else 0,
                "tasks": dict(collections.Counter(r["task"] for r in rows)),
                "source_models": dict(collections.Counter(r["source_model"] for r in rows))}

    report = {"source_trajectories": len(trajs), "filter_counts": dict(counts),
              "candidates": {k: len(v) for k, v in labelled.items()},
              "pairs": len(pairs), "unmatched_bad": unmatched,
              "length_estimate": "tokenizer" if tok else "chars/4",
              "good": stats(good_rows), "bad": stats(bad_rows)}
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("pairs", "candidates", "filter_counts")}, indent=2))
    print(f"Good median tokens {report['good']['median_tokens']}, "
          f"bad median tokens {report['bad']['median_tokens']}")
    print(f"Wrote {out / 'qwen_good_sft.jsonl'} and {out / 'qwen_bad_sft.jsonl'}")
    if len(pairs) < 50:
        print("WARNING: fewer than 50 pairs; the fine-tuned behaviour difference may be weak.")


if __name__ == "__main__":
    main()
