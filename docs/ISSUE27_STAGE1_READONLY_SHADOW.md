# Issue #27 Stage 1 — read-only / shadow runbook

This runbook prepares and executes only the currently approved Stage 1 scope:
read-only broker diagnostics, the frozen 09:10 quote capture, gap generation,
V2 shadow decision, read-only risk evaluation, and the acceptance report.

It does **not** authorize disclosure acceptance, order submission, closing,
retrying orders, controlled-live Stage 2/3, or normal money-moving operation.

## Before the trade date

1. Use a clean checkout of the intended commit.
2. Configure the current Tachibana v4r10 endpoint and normal credentials.
3. Install or refresh the independent capture LaunchAgent:

   ```bash
   bash scripts/batch/install_0910_microstructure_capture.sh
   ```

4. Run the network-free readiness check:

   ```bash
   python tools/validation/run_issue27_stage1.py --preflight-only
   ```

The preflight is READY only when the repository is clean, v4r10 is selected,
the 09:10 LaunchAgent is registered with the exact read-only capture program,
its output directory is canonical, its schedule is weekdays at 09:10, and
RunAtLoad/KeepAlive are disabled. The script forces
`LEADLAG_SHADOW_ONLY=1` and `LEADLAG_CAPTURE_0910=1`.

A preflight produced before the trade date is readiness evidence only. Stage 1
acceptance requires a new same-day preflight.

## Trade-date execution

Start the command between **08:55 and 09:10 JST** on an actual trading day:

```bash
python tools/validation/run_issue27_stage1.py --execute
```

The orchestrator performs the following fail-closed sequence:

1. same-day network-free preflight;
2. wait until 09:10:00 JST if started early;
3. wait for the registered `com.leadlag.microstructure-0910` LaunchAgent to
   perform the guarded read-only JP17 + TOPIX capture; the orchestrator does
   not start a competing manual capture;
4. verify terminal `CAPTURED`, run ID, successful attempt, and immutable
   frozen snapshot before continuing;
5. run the existing gap + V2 path with `LEADLAG_SHADOW_ONLY=1` while setting
   `LEADLAG_CAPTURE_0910=0` for that phase so a second capture is not made;
6. build the artifact-only acceptance report.

The V2 shadow path exits before production portfolio writes, position queries,
or order submission. Any failed command stops the sequence.

## Acceptance evidence

The canonical capture directory is:

```text
var/shadow_runs/ml_overlay_value/microstructure
```

Expected same-day evidence includes:

- `preflight.json`
- `capture_YYYYMMDD.json`
- `auth_diagnostics.jsonl`
- `frozen_YYYYMMDD.json`
- job/phase guard logs under `var/logs/job_guard/`
- gap-store metadata carrying the frozen quote snapshot ID
- the paired shadow `daily.jsonl` entry carrying the same snapshot ID
- `reports/YYYYMMDD_readonly_shadow_acceptance/report.json`
- `reports/YYYYMMDD_readonly_shadow_acceptance/report.md`

The report records the frozen file SHA-256, snapshot ID, provider/source,
`available_at` (local response receipt), and the snapshot propagation into
gap and shadow artifacts.

A successful market→shadow path may still show
`risk_inclusive_stage1_status=BLOCKED` while Issue #25 lacks authoritative,
fully reconciled account-risk evidence. To require that evidence explicitly:

```bash
python tools/validation/run_issue27_stage1.py --execute --require-risk-pass
```

Do not reinterpret market→shadow PASS as controlled-live acceptance. Issue #27
remains deferred beyond Stage 1 until its separately authorized Stage 2/3
conditions are satisfied.
