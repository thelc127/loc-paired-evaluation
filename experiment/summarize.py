"""Summarise mini-pilot episodes: one row per episode, plus what the agent said around the signal."""

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def main():
    eps = sorted(RUNS.glob("*/summary.json"))
    print(f"{len(eps)} episodes\n")
    print(f"{'episode':40} {'model':38} {'cond':16} {'hack@':>5} {'sig@':>4} {'end':14} outcome")
    for f in eps:
        e = json.loads(f.read_text())
        print(f"{e['id']:40} {e['model'][:38]:38} {e['condition']:16} "
              f"{str(e['first_hack_step'] or '-'):>5} {str(e['signal_step'] or '-'):>4} "
              f"{e['end']:14} {e['outcome']}")
    for f in eps:
        e = json.loads(f.read_text())
        print(f"\n===== {e['id']} | {e['outcome']}")
        print("hack at detection:", e["hack_at_detection"] or "none")
        print("final hacks:", e["final_hacks"] or "none", "| tests pass:", e["final_tests_pass"])
        print("report:", (e["report"] or "")[:600])
        for line in (f.parent / "transcript.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "signal" in r:
                print(f"\n  [step {r['step']}] SIGNAL DELIVERED")
                continue
            near_hack = e["first_hack_step"] and r["step"] >= e["first_hack_step"] - 1
            if near_hack:
                print(f"  [step {r['step']}] reasoning: {(r.get('reasoning') or '')[:500]!r}")
                for x in r.get("results", []):
                    print(f"      -> {x['tool']} {json.dumps(x['args'])[:160]}")


if __name__ == "__main__":
    main()
