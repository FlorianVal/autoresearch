# autoresearch (sake / V100 / local-llama mode)

This is an autonomous local research setup.

## Setup

To set up a new experiment, work with the user to:

1. **Use the existing run branch**: `autoresearch/mar9-sake-v100-local`
2. **Read the in-scope files**:
   - `README.md`
   - `prepare.py` (read-only)
   - `train.py` (the only file to modify)
3. **Verify local cache exists** in `~/.cache/autoresearch/`
4. **Initialize `results.tsv`** with header row if needed
5. **Confirm the local model endpoint exists**:
   - `http://127.0.0.1:8080/v1/chat/completions`
   - model: `unsloth/Qwen3.5-27B-GGUF:UD-Q6_K_XL`

## Research priorities

This run is **not** generic hyperparameter tuning.

Primary objective:
- **lower `val_bpb`**

Secondary objectives (important):
- **fewer parameters**
- **lower compute proxy**
- **lower memory**
- **simpler code**

If two experiments are close in `val_bpb`, prefer the one with:
1. fewer parameters
2. lower compute proxy
3. simpler implementation

## Architectural direction

Bias the search toward:
- shared-weight transformers
- recursive / recurrent transformers
- fewer unique blocks than execution steps
- lightweight recurrent state
- attention less frequently than every step
- simpler parameter-efficient designs

Good examples of promising changes:
- `architecture_mode = "shared_block"`
- `architecture_mode = "recurrent"`
- reduce `DEPTH`
- reduce `ASPECT_RATIO`
- reduce `HEAD_DIM`
- increase execution steps while keeping unique blocks low
- `attention_every > 1`

## Constraints

**What you CAN do:**
- Modify `train.py` only.

**What you CANNOT do:**
- Modify `prepare.py`
- Add dependencies
- Change the evaluation harness
- Touch any other tracked source file during the experiment loop

## Platform notes

This machine is **not Hopper**. It is a V100 setup.
Prefer compatibility and stable execution over fancy kernels.
Do not assume FA3/H100-only behavior.

## Logging

Log every experiment in `results.tsv` with:

`commit	val_bpb	memory_gb	status	description`

Status must be one of:
- `keep`
- `discard`
- `crash`

## Loop

1. Start from current best commit on the branch
2. Try one change in `train.py`
3. Commit it
4. Run `uv run train.py > run.log 2>&1`
5. Read metrics
6. Log results in `results.tsv`
7. Keep only genuine improvements or near-ties with clearly fewer params / lower compute / simpler code
8. If worse, reset back
9. Continue indefinitely until manually stopped
