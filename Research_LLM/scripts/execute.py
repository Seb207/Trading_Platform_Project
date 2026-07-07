#!/usr/bin/env python3
"""
Harness phase/step bookkeeping tool — semi-automated.

Unlike a fully autonomous runner, this does NOT invoke Claude itself. A
live Claude Code session implements each step interactively; this script
only handles the bookkeeping around that: scaffolding phase files, loading
guardrails (CLAUDE.md + module CLAUDE.md + docs/), tracking step status,
and the git branch/commit automation.

Usage:
    python3 scripts/execute.py new   <phase-dir> <step-name> [<step-name> ...]
    python3 scripts/execute.py start <phase-dir>
    python3 scripts/execute.py next  <phase-dir>
    python3 scripts/execute.py done  <phase-dir> <step-num> <summary>
    python3 scripts/execute.py fail  <phase-dir> <step-num> <error-message>
    python3 scripts/execute.py block <phase-dir> <step-num> <reason>
    python3 scripts/execute.py status
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"
TOP_INDEX = PHASES_DIR / "index.json"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _phase_dir(name: str) -> Path:
    return PHASES_DIR / name


def _phase_index(name: str) -> Path:
    return _phase_dir(name) / "index.json"


def _load_guardrails() -> str:
    """Root CLAUDE.md + every direct-child module's CLAUDE.md + docs/*.md.

    Deliberately excludes Research_LLM/CLAUDE.md — that directory is a
    separate git repository with its own copy of this script and its own
    guardrail set (see docs/ARCHITECTURE.md).
    """
    sections = []
    root_claude = ROOT / "CLAUDE.md"
    if root_claude.exists():
        sections.append(f"## Root CLAUDE.md\n\n{root_claude.read_text()}")
    for child in sorted(ROOT.iterdir()):
        if child.name == "Research_LLM" or not child.is_dir():
            continue
        module_claude = child / "CLAUDE.md"
        if module_claude.exists():
            sections.append(f"## {child.name}/CLAUDE.md\n\n{module_claude.read_text()}")
    docs_dir = ROOT / "docs"
    if docs_dir.is_dir():
        for doc in sorted(docs_dir.glob("*.md")):
            sections.append(f"## docs/{doc.name}\n\n{doc.read_text()}")
    return "\n\n---\n\n".join(sections)


def _update_top_index(phase_name: str, status: str) -> None:
    if not TOP_INDEX.exists():
        return
    top = _read_json(TOP_INDEX)
    ts = _stamp()
    for phase in top.get("phases", []):
        if phase.get("dir") == phase_name:
            phase["status"] = status
            key = {"completed": "completed_at", "error": "failed_at", "blocked": "blocked_at"}.get(status)
            if key:
                phase[key] = ts
            break
    _write_json(TOP_INDEX, top)


# ── Commands ─────────────────────────────────────────────────────────────

def cmd_new(args) -> None:
    phase_name = args.phase_dir
    step_names = args.steps
    if not step_names:
        print("ERROR: provide at least one step name")
        sys.exit(1)

    PHASES_DIR.mkdir(exist_ok=True)
    phase_dir = _phase_dir(phase_name)
    if phase_dir.exists():
        print(f"ERROR: {phase_dir} already exists")
        sys.exit(1)
    phase_dir.mkdir(parents=True)

    steps = [{"step": i + 1, "name": name, "status": "pending"} for i, name in enumerate(step_names)]
    _write_json(phase_dir / "index.json", {
        "project": ROOT.name,
        "phase": phase_name,
        "created_at": _stamp(),
        "steps": steps,
    })

    for i, name in enumerate(step_names, start=1):
        step_file = phase_dir / f"step{i}.md"
        step_file.write_text(
            f"# Step {i}: {name}\n\n"
            f"## What to build\n\nTODO — fill in per harness.md's Phase C principles.\n\n"
            f"## Files touched\n\nTODO\n\n"
            f"## Why (one line)\n\nTODO\n\n"
            f"## Acceptance Criteria\n\n```bash\nTODO — an executable check\n```\n",
            encoding="utf-8",
        )

    if not TOP_INDEX.exists():
        _write_json(TOP_INDEX, {"phases": []})
    top = _read_json(TOP_INDEX)
    top.setdefault("phases", []).append({"dir": phase_name, "status": "pending", "created_at": _stamp()})
    _write_json(TOP_INDEX, top)

    print(f"Created phases/{phase_name}/ with {len(step_names)} step file(s). Fill in each step<N>.md, then:")
    print(f"  python3 scripts/execute.py start {phase_name}")


def cmd_start(args) -> None:
    phase_name = args.phase_dir
    if not _phase_index(phase_name).exists():
        print(f"ERROR: {_phase_index(phase_name)} not found — run 'new' first")
        sys.exit(1)

    branch = f"feat-{phase_name}"
    r = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    if r.returncode != 0:
        print("ERROR: not a git repo, or git unavailable.")
        print(r.stderr.strip())
        sys.exit(1)
    if r.stdout.strip() == branch:
        print(f"Already on {branch}")
        return

    r = _run_git("rev-parse", "--verify", branch)
    r = _run_git("checkout", branch) if r.returncode == 0 else _run_git("checkout", "-b", branch)
    if r.returncode != 0:
        print(f"ERROR: could not check out '{branch}'.")
        print(r.stderr.strip())
        print("Hint: stash or commit pending changes first.")
        sys.exit(1)
    print(f"Branch: {branch}")


def cmd_next(args) -> None:
    phase_name = args.phase_dir
    index = _read_json(_phase_index(phase_name))
    pending = next((s for s in index["steps"] if s["status"] == "pending"), None)

    completed = [s for s in index["steps"] if s["status"] == "completed"]
    if completed:
        print("## Completed so far\n")
        for s in completed:
            print(f"- Step {s['step']} ({s['name']}): {s.get('summary', '')}")
        print()

    if pending is None:
        blocked = [s for s in index["steps"] if s["status"] in ("error", "blocked")]
        if blocked:
            s = blocked[0]
            print(f"Step {s['step']} ({s['name']}) is '{s['status']}': "
                  f"{s.get('error_message') or s.get('blocked_reason', '')}")
            print("Resolve it and reset its status to 'pending' to retry.")
            sys.exit(1)
        print(f"All steps in '{phase_name}' completed.")
        return

    step_file = _phase_dir(phase_name) / f"step{pending['step']}.md"
    print(f"## Guardrails\n\n{_load_guardrails()}\n")
    print(f"## Step {pending['step']}/{len(index['steps'])}: {pending['name']}\n")
    print(step_file.read_text() if step_file.exists() else "(step file missing)")


def _set_step(phase_name: str, step_num: int, **fields) -> dict:
    index = _read_json(_phase_index(phase_name))
    target = None
    for s in index["steps"]:
        if s["step"] == step_num:
            s.update(fields)
            target = s
            break
    if target is None:
        print(f"ERROR: step {step_num} not found in phase '{phase_name}'")
        sys.exit(1)
    _write_json(_phase_index(phase_name), index)
    return index


def cmd_done(args) -> None:
    phase_name, step_num, summary = args.phase_dir, args.step_num, args.summary
    _set_step(phase_name, step_num, status="completed", summary=summary, completed_at=_stamp())

    step_name = next(s["name"] for s in _read_json(_phase_index(phase_name))["steps"] if s["step"] == step_num)
    _run_git("add", "-A")
    if _run_git("diff", "--cached", "--quiet").returncode != 0:
        msg = f"feat({phase_name}): step {step_num} — {step_name}"
        r = _run_git("commit", "-m", msg)
        if r.returncode == 0:
            print(f"Commit: {msg}")
        else:
            print(f"WARN: commit failed: {r.stderr.strip()}")
    print(f"Step {step_num} ({step_name}) marked completed.")

    index = _read_json(_phase_index(phase_name))
    if all(s["status"] == "completed" for s in index["steps"]):
        index["completed_at"] = _stamp()
        _write_json(_phase_index(phase_name), index)
        _update_top_index(phase_name, "completed")
        print(f"Phase '{phase_name}' complete.")


def cmd_fail(args) -> None:
    _set_step(args.phase_dir, args.step_num, status="error", error_message=args.message, failed_at=_stamp())
    _update_top_index(args.phase_dir, "error")
    print(f"Step {args.step_num} marked error. Fix and reset status to 'pending' to retry.")


def cmd_block(args) -> None:
    _set_step(args.phase_dir, args.step_num, status="blocked", blocked_reason=args.reason, blocked_at=_stamp())
    _update_top_index(args.phase_dir, "blocked")
    print(f"Step {args.step_num} marked blocked: {args.reason}")
    print("This usually means a credential or local service the user needs to provide — ask them, don't improvise.")


def cmd_status(_args) -> None:
    if not TOP_INDEX.exists():
        print("No phases yet — run 'new' to create one.")
        return
    top = _read_json(TOP_INDEX)
    for phase in top.get("phases", []):
        print(f"- {phase['dir']}: {phase['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness phase/step bookkeeping tool (semi-automated)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="Scaffold a new phase")
    p_new.add_argument("phase_dir")
    p_new.add_argument("steps", nargs="+")
    p_new.set_defaults(func=cmd_new)

    p_start = sub.add_parser("start", help="Check out the phase's feature branch")
    p_start.add_argument("phase_dir")
    p_start.set_defaults(func=cmd_start)

    p_next = sub.add_parser("next", help="Show the next pending step + guardrails")
    p_next.add_argument("phase_dir")
    p_next.set_defaults(func=cmd_next)

    p_done = sub.add_parser("done", help="Mark a step completed and commit it")
    p_done.add_argument("phase_dir")
    p_done.add_argument("step_num", type=int)
    p_done.add_argument("summary")
    p_done.set_defaults(func=cmd_done)

    p_fail = sub.add_parser("fail", help="Mark a step as failed")
    p_fail.add_argument("phase_dir")
    p_fail.add_argument("step_num", type=int)
    p_fail.add_argument("message")
    p_fail.set_defaults(func=cmd_fail)

    p_block = sub.add_parser("block", help="Mark a step as blocked on user input")
    p_block.add_argument("phase_dir")
    p_block.add_argument("step_num", type=int)
    p_block.add_argument("reason")
    p_block.set_defaults(func=cmd_block)

    p_status = sub.add_parser("status", help="List all phases and their status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
