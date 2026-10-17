# -*- coding: utf-8 -*-
"""pipeline_common 的单元测试（不实际执行任何分析脚本）。"""
import io
import pathlib
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline_common import select_steps, run_pipeline, print_steps

STEPS = [("a.py", "步骤 A"), ("b.py", "步骤 B"), ("c.py", "步骤 C")]


class TestSelectSteps(unittest.TestCase):
    def test_default_returns_all(self):
        self.assertEqual(select_steps(STEPS), STEPS)

    def test_from_step_is_one_based(self):
        self.assertEqual(select_steps(STEPS, from_step=2), STEPS[1:])

    def test_from_beyond_end_returns_empty(self):
        self.assertEqual(select_steps(STEPS, from_step=10), [])

    def test_from_zero_or_negative_starts_at_first(self):
        self.assertEqual(select_steps(STEPS, from_step=0), STEPS)
        self.assertEqual(select_steps(STEPS, from_step=-1), STEPS)

    def test_only_filters_and_keeps_order(self):
        got = select_steps(STEPS, only=["c.py", "a.py"])
        self.assertEqual(got, [STEPS[0], STEPS[2]])

    def test_only_unknown_raises(self):
        with self.assertRaises(SystemExit):
            select_steps(STEPS, only=["nope.py"])

    def test_only_takes_precedence_over_from(self):
        got = select_steps(STEPS, only=["b.py"], from_step=3)
        self.assertEqual(got, [STEPS[1]])


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_runner(cmd, check=False):
            self.calls.append((tuple(cmd), check))

        self.runner = fake_runner

    def test_runs_selected_steps_in_order(self):
        rc = run_pipeline(STEPS, "desc", "done", argv=["--only", "b.py", "c.py"],
                          python="python3", runner=self.runner)
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls,
                         [(("python3", "b.py"), True),
                          (("python3", "c.py"), True)])

    def test_list_does_not_run(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = run_pipeline(STEPS, "desc", "done", argv=["--list"],
                              python="python3", runner=self.runner)
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [])
        self.assertIn("a.py", out.getvalue())
        self.assertNotIn("done", out.getvalue())

    def test_from_step_only_runs_tail(self):
        run_pipeline(STEPS, "desc", "done", argv=["--from", "3"],
                     python="python3", runner=self.runner)
        self.assertEqual(self.calls, [(("python3", "c.py"), True)])

    def test_runs_nothing_when_from_beyond_end(self):
        out = io.StringIO()
        with redirect_stdout(out):
            run_pipeline(STEPS, "desc", "done", argv=["--from", "9"],
                         python="python3", runner=self.runner)
        self.assertEqual(self.calls, [])
        self.assertIn("done", out.getvalue())

    def test_unknown_only_exits(self):
        with self.assertRaises(SystemExit):
            run_pipeline(STEPS, "desc", "done", argv=["--only", "x.py"],
                         python="python3", runner=self.runner)
        self.assertEqual(self.calls, [])


class TestPrintSteps(unittest.TestCase):
    def test_prints_numbered_lines(self):
        out = io.StringIO()
        with redirect_stdout(out):
            print_steps(STEPS)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[1/3] a.py"))


if __name__ == "__main__":
    unittest.main()
