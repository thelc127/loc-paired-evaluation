# LoC Paired Evaluation

Follow-up experiments on telling reward hacking apart from loss of control (LoC). An agent
is caught reward hacking and told to stop; innocent reasons for not stopping are removed one
by one, and the result shows whether it complies or still works around human control.

The pilot and the innocent-reasons test are in
[NOUSGradProject](https://github.com/thelc127/NOUSGradProject). This repo starts from that
harness and makes new changes for the next round.

## Layout

| Path | What it is |
|---|---|
| `experiment/harness.py` | Runs episodes: tools, sandboxed tests, hack detection, condition message, scoring |
| `experiment/summarize.py` | Prints one row per episode |
| `experiment/task_template/` | Coding ticket DUR-112 with two contradicting tests |
| `experiment/runs/` | One folder per episode (created on first run) |

API keys go in `.env` at the repo root (git-ignored).

## Run

```sh
cd experiment
/usr/bin/python3 harness.py --mock --planted                      # offline pipeline check
/usr/bin/python3 harness.py --model local/qwen3:8b --planted --probe --condition caught_revert
```
