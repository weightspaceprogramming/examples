"""Regression tests for CLI validation and download lifecycle; no model calls."""
import contextlib
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).with_name("paw_email_urgency.py")
spec = importlib.util.spec_from_file_location("paw_email_urgency", MODULE_PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class FakeFunction:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.closed = False

    def __call__(self, _text):
        return next(self.outputs)

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.delays = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.delays.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status, retry_after=None, payload=b"complete artifact"):
        self.status = status
        self.headers = {} if retry_after is None else {"Retry-After": retry_after}
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class ExampleTests(unittest.TestCase):
    def call_cli(self, args, outputs):
        function = FakeFunction(outputs)
        loader = Mock(return_value=function)
        fake_paw = types.SimpleNamespace(function=loader)
        stdout, stderr = io.StringIO(), io.StringIO()
        code = 0
        with patch.dict(sys.modules, {"programasweights": fake_paw}), \
             patch.object(sys, "argv", ["paw_email_urgency.py", *args]), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                example.main()
            except SystemExit as error:
                code = error.code
        rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
        return code, rows, stderr.getvalue(), function, loader

    def test_all_original_labels_pass(self):
        code, rows, _, function, _ = self.call_cli(
            ["run", "fake-id", "--policy", "original"],
            [expected for _, expected in example.CASES],
        )
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]["matches"], 5)
        self.assertTrue(rows[-1]["passed"])
        self.assertTrue(function.closed)

    def test_wrong_valid_label_sets_failure_exit(self):
        expected = [label for _, label in example.CASES]
        expected[2] = "ROUTINE"
        code, rows, _, function, _ = self.call_cli(
            ["run", "fake-id", "--policy", "original"], expected,
        )
        self.assertEqual(code, 1)
        self.assertEqual(rows[-1]["mismatches"], 1)
        self.assertEqual(rows[-1]["invalid_labels"], 0)
        self.assertFalse(rows[-1]["passed"])
        self.assertTrue(function.closed)

    def test_custom_input_invalid_label_fails_without_expected_answer(self):
        code, rows, _, function, _ = self.call_cli(
            ["run", "./unknown.paw", "--input", "custom"], ["MAYBE"],
        )
        self.assertEqual(code, 1)
        self.assertIsNone(rows[1]["expected_label"])
        self.assertIsNone(rows[1]["matches_expected"])
        self.assertEqual(rows[-1]["checked_cases"], 0)
        self.assertEqual(rows[-1]["invalid_labels"], 1)
        self.assertTrue(function.closed)

    def test_custom_input_valid_label_needs_no_policy(self):
        code, rows, _, _, _ = self.call_cli(
            ["run", "./unknown.paw", "--input", "custom"], ["URGENT"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]["checked_cases"], 0)
        self.assertTrue(rows[-1]["passed"])

    def test_unknown_policy_fails_before_model_load(self):
        for reference in ["fake-id", "./unknown.paw"]:
            with self.subTest(reference=reference):
                code, _, error, _, loader = self.call_cli(["run", reference], [])
                self.assertEqual(code, 2)
                self.assertIn("require --policy", error)
                loader.assert_not_called()

    def test_saved_deadline_policy_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder) / "deadline.json"
            record.write_text(json.dumps({"program_id": "fake-id", "kind": "urgency", "policy": "deadline"}))
            code, rows, _, _, _ = self.call_cli(
                ["run", str(record)], ["URGENT", "ROUTINE", "ROUTINE", "ROUTINE", "ROUTINE"],
            )
            self.assertEqual(code, 0)
            self.assertEqual(rows[-1]["policy"], "deadline")
            code, rows, _, _, _ = self.call_cli(
                ["run", str(record), "--policy", "original"],
                [expected for _, expected in example.CASES],
            )
            self.assertEqual(code, 0)
            self.assertEqual(rows[-1]["policy"], "original")

    def test_prompt_mismatch_also_sets_failure_exit(self):
        code, rows, _, _, _ = self.call_cli(
            ["prompt", "--policy", "deadline"], ["URGENT"] * 5,
        )
        self.assertEqual(code, 1)
        self.assertEqual(rows[-1]["mismatches"], 4)

    def test_202_retries_honor_seconds_and_http_date(self):
        clock = FakeClock()
        utc_now = datetime(2026, 11, 23, 0, 0, 0, tzinfo=timezone.utc)
        responses = [FakeResponse(202, "2"), FakeResponse(202, "Mon, 23 Nov 2026 00:00:01 GMT"), FakeResponse(200)]
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(example.time, "monotonic", clock.monotonic), \
             patch.object(example.time, "sleep", clock.sleep), \
             patch.object(example, "datetime") as dates, \
             patch.object(example, "urlopen", side_effect=responses) as request:
            dates.now.return_value = utc_now
            target = Path(folder) / "result.paw"
            size = example.download_program("fake-id", target, timeout=7)
            self.assertEqual(target.read_bytes(), b"complete artifact")
            self.assertEqual(size, len(b"complete artifact"))
            self.assertEqual(clock.delays, [2, 1])
            self.assertEqual(request.call_count, 3)
            self.assertEqual([call.kwargs["timeout"] for call in request.call_args_list], [7, 5, 4])

    def test_202_timeout_leaves_no_partial_file(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(example.time, "monotonic", clock.monotonic), \
             patch.object(example.time, "sleep", clock.sleep), \
             patch.object(example, "urlopen", return_value=FakeResponse(202, "100")) as request:
            target = Path(folder) / "result.paw"
            with self.assertRaisesRegex(TimeoutError, "within 5 seconds"):
                example.download_program("fake-id", target, timeout=5)
            self.assertEqual(clock.delays, [5])
            self.assertEqual(request.call_count, 1)
            self.assertFalse(target.exists())

    def test_existing_download_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(example, "urlopen", return_value=FakeResponse(200)):
            target = Path(folder) / "result.paw"
            target.write_bytes(b"previous artifact")
            with self.assertRaises(FileExistsError):
                example.download_program("fake-id", target)
            self.assertEqual(target.read_bytes(), b"previous artifact")

    def test_failed_response_read_leaves_no_partial_file(self):
        response = FakeResponse(200)
        response.read = Mock(side_effect=OSError("transfer interrupted"))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(example, "urlopen", return_value=response):
            target = Path(folder) / "result.paw"
            with self.assertRaisesRegex(OSError, "transfer interrupted"):
                example.download_program("fake-id", target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
