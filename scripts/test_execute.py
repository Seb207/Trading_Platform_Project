#!/usr/bin/env python3
"""
Tests for execute.py (the harness phase/step bookkeeping tool).

Run: python3 -m unittest scripts.test_execute -v   (from the repo root)
or:  python3 scripts/test_execute.py

Uses only the standard library (unittest + mock) — no pytest dependency,
matching execute.py's own zero-new-deps design.
"""
import argparse
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import execute  # noqa: E402


class HarnessTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_root = execute.ROOT
        self._orig_phases_dir = execute.PHASES_DIR
        self._orig_top_index = execute.TOP_INDEX

        execute.ROOT = Path(self._tmpdir)
        execute.PHASES_DIR = execute.ROOT / "phases"
        execute.TOP_INDEX = execute.PHASES_DIR / "index.json"

    def tearDown(self):
        execute.ROOT = self._orig_root
        execute.PHASES_DIR = self._orig_phases_dir
        execute.TOP_INDEX = self._orig_top_index
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestUtilities(HarnessTestCase):
    def test_stamp_is_iso_utc(self):
        ts = execute._stamp()
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_json_round_trip(self):
        p = execute.ROOT / "sample.json"
        execute.ROOT.mkdir(parents=True, exist_ok=True)
        data = {"a": 1, "b": [1, 2, 3], "unicode": "café résumé"}
        execute._write_json(p, data)
        self.assertEqual(execute._read_json(p), data)


class TestGuardrails(HarnessTestCase):
    def test_loads_root_and_module_claude_but_not_research_llm(self):
        execute.ROOT.mkdir(parents=True, exist_ok=True)
        (execute.ROOT / "CLAUDE.md").write_text("ROOT RULES")

        module_dir = execute.ROOT / "Some Module"
        module_dir.mkdir()
        (module_dir / "CLAUDE.md").write_text("MODULE RULES")

        research_llm = execute.ROOT / "Research_LLM"
        research_llm.mkdir()
        (research_llm / "CLAUDE.md").write_text("SHOULD NOT APPEAR")

        docs_dir = execute.ROOT / "docs"
        docs_dir.mkdir()
        (docs_dir / "ADR.md").write_text("SOME ADR")

        guardrails = execute._load_guardrails()
        self.assertIn("ROOT RULES", guardrails)
        self.assertIn("MODULE RULES", guardrails)
        self.assertIn("SOME ADR", guardrails)
        self.assertNotIn("SHOULD NOT APPEAR", guardrails)

    def test_missing_claude_md_does_not_raise(self):
        execute.ROOT.mkdir(parents=True, exist_ok=True)
        self.assertEqual(execute._load_guardrails(), "")


class TestPhaseLifecycle(HarnessTestCase):
    def _new_args(self, phase="0-mvp", steps=("build-api", "build-ui")):
        return argparse.Namespace(phase_dir=phase, steps=list(steps))

    def test_new_creates_index_and_step_files(self):
        execute.cmd_new(self._new_args())

        index = execute._read_json(execute._phase_index("0-mvp"))
        self.assertEqual(len(index["steps"]), 2)
        self.assertEqual(index["steps"][0]["status"], "pending")
        self.assertTrue((execute._phase_dir("0-mvp") / "step1.md").exists())
        self.assertTrue((execute._phase_dir("0-mvp") / "step2.md").exists())

        top = execute._read_json(execute.TOP_INDEX)
        self.assertEqual(top["phases"][0]["dir"], "0-mvp")
        self.assertEqual(top["phases"][0]["status"], "pending")

    def test_new_refuses_to_overwrite_existing_phase(self):
        execute.cmd_new(self._new_args())
        with self.assertRaises(SystemExit):
            execute.cmd_new(self._new_args())

    @patch("execute._run_git")
    def test_done_marks_completed_and_commits(self, mock_git):
        mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
        execute.cmd_new(self._new_args())

        args = argparse.Namespace(phase_dir="0-mvp", step_num=1, summary="did the thing")
        execute.cmd_done(args)

        index = execute._read_json(execute._phase_index("0-mvp"))
        step1 = next(s for s in index["steps"] if s["step"] == 1)
        self.assertEqual(step1["status"], "completed")
        self.assertEqual(step1["summary"], "did the thing")
        self.assertIn("completed_at", step1)

    @patch("execute._run_git")
    def test_done_on_last_step_completes_phase(self, mock_git):
        mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
        execute.cmd_new(self._new_args(steps=("only-step",)))

        execute.cmd_done(argparse.Namespace(phase_dir="0-mvp", step_num=1, summary="done"))

        top = execute._read_json(execute.TOP_INDEX)
        self.assertEqual(top["phases"][0]["status"], "completed")

    def test_fail_marks_error_with_message(self):
        execute.cmd_new(self._new_args())
        execute.cmd_fail(argparse.Namespace(phase_dir="0-mvp", step_num=1, message="build broke"))

        index = execute._read_json(execute._phase_index("0-mvp"))
        step1 = next(s for s in index["steps"] if s["step"] == 1)
        self.assertEqual(step1["status"], "error")
        self.assertEqual(step1["error_message"], "build broke")

    def test_block_marks_blocked_with_reason(self):
        execute.cmd_new(self._new_args())
        execute.cmd_block(argparse.Namespace(phase_dir="0-mvp", step_num=1, reason="needs OpenRouter API key"))

        index = execute._read_json(execute._phase_index("0-mvp"))
        step1 = next(s for s in index["steps"] if s["step"] == 1)
        self.assertEqual(step1["status"], "blocked")
        self.assertEqual(step1["blocked_reason"], "needs OpenRouter API key")

        top = execute._read_json(execute.TOP_INDEX)
        self.assertEqual(top["phases"][0]["status"], "blocked")

    def test_next_reports_all_completed(self):
        execute.cmd_new(self._new_args(steps=("only-step",)))
        with patch("execute._run_git", return_value=MagicMock(returncode=0, stdout="", stderr="")):
            execute.cmd_done(argparse.Namespace(phase_dir="0-mvp", step_num=1, summary="done"))

        with patch("builtins.print") as mock_print:
            execute.cmd_next(argparse.Namespace(phase_dir="0-mvp"))
            output = "\n".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
            self.assertIn("completed", output.lower())


class TestGitBranching(HarnessTestCase):
    @patch("execute._run_git")
    def test_start_checks_out_new_branch(self, mock_git):
        execute.cmd_new(argparse.Namespace(phase_dir="0-mvp", steps=["step-one"]))

        mock_git.side_effect = [
            MagicMock(returncode=0, stdout="main\n", stderr=""),   # current branch
            MagicMock(returncode=1, stdout="", stderr=""),          # branch doesn't exist yet
            MagicMock(returncode=0, stdout="", stderr=""),          # checkout -b succeeds
        ]
        execute.cmd_start(argparse.Namespace(phase_dir="0-mvp"))

        checkout_calls = [c for c in mock_git.call_args_list if "checkout" in c.args]
        self.assertTrue(any("-b" in c.args for c in checkout_calls))

    @patch("execute._run_git")
    def test_start_exits_if_not_a_git_repo(self, mock_git):
        execute.cmd_new(argparse.Namespace(phase_dir="0-mvp", steps=["step-one"]))
        mock_git.return_value = MagicMock(returncode=128, stdout="", stderr="not a git repository")

        with self.assertRaises(SystemExit):
            execute.cmd_start(argparse.Namespace(phase_dir="0-mvp"))


if __name__ == "__main__":
    unittest.main()
