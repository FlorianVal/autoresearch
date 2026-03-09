#!/usr/bin/env python3
"""Continuous local autoresearch loop using the local llama.cpp OpenAI-compatible server.
It edits train.py only, runs experiments, keeps improvements, and logs to results.tsv.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request
from typing import Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
PROGRAM = ROOT / "program.md"
RESULTS = ROOT / "results.tsv"
RUN_LOG = ROOT / "run.log"
MODEL = os.environ.get("AUTORESEARCH_LOCAL_MODEL", "unsloth/Qwen3.5-27B-GGUF:UD-Q6_K_XL")
API_URL = os.environ.get("AUTORESEARCH_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
MAX_ITERS = int(os.environ.get("AUTORESEARCH_MAX_ITERS", "0"))  # 0 = forever
SLEEP_ON_ERROR = int(os.environ.get("AUTORESEARCH_SLEEP_ON_ERROR", "30"))

SYSTEM = """You are the local autoresearch coding agent.\nYou are optimizing a single file train.py for lower val_bpb on a fixed 5-minute time budget.\nYou MUST prefer lower-parameter and lower-compute designs when loss is similar.\nYou MUST focus on recursive/shared-weight transformers, sparse attention frequency, and simplification.\nReturn ONLY valid JSON with keys: description, train_py. No markdown fences.\nThe train_py value must contain the complete replacement content of train.py.\nKeep the file runnable. Do not modify any file other than train.py.\n"""


def sh(cmd: str, timeout: Optional[int] = None, check: bool = True) -> str:
    p = subprocess.run(cmd, shell=True, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout.strip()


def get_commit() -> str:
    return sh("git rev-parse --short HEAD")


def git_commit(msg: str) -> str:
    sh("git add train.py")
    # if nothing changed, git commit exits non-zero
    p = subprocess.run(f"git commit -m {json.dumps(msg)}", shell=True, cwd=ROOT, text=True, capture_output=True)
    if p.returncode != 0:
        if "nothing to commit" in (p.stdout + p.stderr):
            return get_commit()
        raise RuntimeError(p.stdout + "\n" + p.stderr)
    return get_commit()


def git_reset(commit: str):
    sh(f"git reset --hard {commit}")


def parse_metrics(text: str) -> Optional[dict]:
    out = {}
    for key in ["val_bpb", "peak_vram_mb", "num_params_M", "compute_proxy", "architecture"]:
        m = re.search(rf"^{re.escape(key)}:\s+(.*)$", text, flags=re.M)
        if m:
            out[key] = m.group(1).strip()
    if "val_bpb" not in out:
        return None
    out["val_bpb"] = float(out["val_bpb"])
    out["peak_vram_mb"] = float(out.get("peak_vram_mb", 0.0))
    out["num_params_M"] = float(out.get("num_params_M", 0.0))
    try:
        out["compute_proxy"] = float(out.get("compute_proxy", 0.0))
    except Exception:
        out["compute_proxy"] = 0.0
    return out


def run_experiment() -> Tuple[Optional[dict], str]:
    cmd = "timeout 650s bash -lc 'uv run train.py > run.log 2>&1'"
    subprocess.run(cmd, shell=True, cwd=ROOT)
    text = RUN_LOG.read_text(errors="replace") if RUN_LOG.exists() else ""
    metrics = parse_metrics(text)
    return metrics, text


def ensure_results_header():
    if not RESULTS.exists() or RESULTS.read_text().strip() == "":
        RESULTS.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def append_result(commit: str, metrics: Optional[dict], status: str, description: str):
    ensure_results_header()
    if metrics is None:
        line = f"{commit}\t0.000000\t0.0\t{status}\t{description}\n"
    else:
        mem_gb = metrics["peak_vram_mb"] / 1024.0
        line = f"{commit}\t{metrics['val_bpb']:.6f}\t{mem_gb:.1f}\t{status}\t{description}\n"
    with RESULTS.open("a") as f:
        f.write(line)


def current_best() -> Optional[float]:
    if not RESULTS.exists():
        return None
    best = None
    for line in RESULTS.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        status = parts[3]
        if status != "keep":
            continue
        try:
            val = float(parts[1])
        except Exception:
            continue
        best = val if best is None else min(best, val)
    return best


def tail_results(n: int = 20) -> str:
    if not RESULTS.exists():
        return ""
    lines = RESULTS.read_text().splitlines()
    return "\n".join(lines[-n:])


def query_model(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 12000,
    }
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read().decode())
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")
    m = re.search(r"\{.*\}\s*$", content, flags=re.S)
    raw = m.group(0) if m else content.strip()
    return json.loads(raw)


def build_prompt(last_log_tail: str = "") -> str:
    train_text = TRAIN.read_text()
    program_text = PROGRAM.read_text()
    best = current_best()
    best_str = "none yet" if best is None else f"{best:.6f}"
    return f"""
Repository goal:
- improve val_bpb in 5-minute runs
- prefer fewer parameters and lower compute when loss is similar
- focus on shared-block / recurrent / recursive transformer directions
- keep train.py runnable on a V100 with local SDPA fallback

Current best val_bpb: {best_str}

Recent results:
{tail_results()}

If the previous run crashed, here is the tail of run.log:
{last_log_tail}

program.md:
{program_text}

Current train.py:
{train_text}

Propose ONE experiment only.
Requirements:
- modify train.py only
- keep file self-contained
- if loss gain is tiny, prefer a parameter/computation reduction
- if changing architecture_mode defaults, explain why in description
Return JSON only.
"""


def establish_baseline():
    best = current_best()
    if best is not None:
        return
    commit = get_commit()
    metrics, log = run_experiment()
    if metrics is None:
        append_result(commit, None, "crash", "baseline crash")
        raise RuntimeError("baseline crashed\n" + "\n".join(log.splitlines()[-80:]))
    append_result(commit, metrics, "keep", "baseline")


def main():
    ensure_results_header()
    establish_baseline()
    iterations = 0
    while True:
        if MAX_ITERS and iterations >= MAX_ITERS:
            print("Reached AUTORESEARCH_MAX_ITERS, exiting.")
            return
        start_commit = get_commit()
        try:
            proposal = query_model(build_prompt())
            description = str(proposal.get("description", "local model experiment")).replace("\t", " ").replace("\n", " ")[:200]
            train_py = proposal["train_py"]
            if "val_bpb" not in train_py or "architecture" not in train_py:
                raise RuntimeError("proposal missing expected train.py structure")
            TRAIN.write_text(train_py)
            exp_commit = git_commit(f"autoresearch: {description}")
            metrics, log = run_experiment()
            if metrics is None:
                append_result(exp_commit, None, "crash", description)
                git_reset(start_commit)
            else:
                prev_best = current_best()
                improved = prev_best is None or metrics["val_bpb"] < prev_best - 1e-6
                if not improved and prev_best is not None and abs(metrics["val_bpb"] - prev_best) <= 5e-4:
                    # allow near-ties if they are materially smaller or simpler on proxy metrics
                    best_rows = [ln.split("\t") for ln in RESULTS.read_text().splitlines()[1:] if "\tkeep\t" in ln]
                    best_params = None
                    for row in best_rows:
                        try:
                            if abs(float(row[1]) - prev_best) <= 1e-6:
                                best_params = float(row[2])
                                break
                        except Exception:
                            pass
                    if best_params is None or (metrics["peak_vram_mb"] / 1024.0) <= best_params:
                        improved = True
                status = "keep" if improved else "discard"
                append_result(exp_commit, metrics, status, description)
                if not improved:
                    git_reset(start_commit)
            iterations += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:
            err = f"agent loop error: {e}"
            print(err, file=sys.stderr)
            time.sleep(SLEEP_ON_ERROR)
            try:
                git_reset(start_commit)
            except Exception:
                pass


if __name__ == "__main__":
    main()
