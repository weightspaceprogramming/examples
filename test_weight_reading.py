"""Check tensor layout, input rejection, cache integrity and actor equivalence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

import numpy as np

import weight_reading as reader


class WeightReadingTests(unittest.TestCase):
    def test_known_channel_and_cell_order(self):
        weights = np.zeros((12, 2))
        weights[(1 * 2 + 0) * 3 + 2] = [3, 4]
        maps = reader.channel_norms(weights, (2, 2, 3))
        self.assertEqual(maps.shape, (3, 2, 2))
        self.assertEqual(maps[2, 1, 0], 5)
        self.assertEqual(np.count_nonzero(maps), 1)

    def test_matches_independent_cell_loop(self):
        rng = np.random.default_rng(11)
        weights = rng.normal(size=(30, 7))
        actual = reader.channel_norms(weights, (2, 3, 5))
        expected = np.empty((5, 2, 3))
        for channel in range(5):
            for row in range(2):
                for column in range(3):
                    index = (row * 3 + column) * 5 + channel
                    expected[channel, row, column] = sum(float(x) ** 2 for x in weights[index]) ** 0.5
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_invalid_kernel_and_grid(self):
        cases = [(np.ones((12, 3)), (0, 2, 3)), (np.ones((12, 3)), (2, 2, 2)),
                 (np.ones((12, 0)), (2, 2, 3)), (np.ones(12), (2, 2, 3)),
                 (np.ones((12, 2), dtype=complex), (2, 2, 3)),
                 (np.full((12, 2), np.nan), (2, 2, 3)), (np.full((12, 2), np.inf), (2, 2, 3)),
                 (np.ones((12, 3)), (2, 2, True))]
        for kernel, shape in cases:
            with self.subTest(shape=shape, dtype=kernel.dtype), self.assertRaises(ValueError):
                reader.channel_norms(kernel, shape)

    def test_integer_norms_do_not_overflow(self):
        maps = reader.channel_norms(np.full((1, 2), 200, dtype=np.uint8), (1, 1, 1))
        self.assertAlmostEqual(maps[0, 0, 0], np.sqrt(80000))

    def test_square_accumulation_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError, "Squared kernel magnitudes exceed float64 range"):
            reader.channel_norms(np.full((1, 2), 1e300), (1, 1, 1))

    def test_kernel_and_channel_errors(self):
        with self.assertRaisesRegex(ValueError, "channel label"):
            reader.map_record("test", {"w": np.ones((4, 2))}, "w", (2, 2, 1), [])
        with self.assertRaisesRegex(ValueError, "absent"):
            reader.map_record("test", {}, "w", (2, 2, 1), ["input"])

    def test_cached_bytes_verified_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / reader.REVISION
            directory.mkdir()
            (directory / "test.npz").write_bytes(b"verified")
            with patch.object(reader.urllib.request, "urlopen", side_effect=AssertionError("network call")):
                path, _ = reader.download_network("test", reader.sha256(b"verified"), temp)
                self.assertEqual(path.read_bytes(), b"verified")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    reader.download_network("test", reader.sha256(b"different"), temp)
                self.assertEqual(path.read_bytes(), b"verified")

    def test_bad_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(reader.urllib.request, "urlopen", return_value=io.BytesIO(b"bad download")):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    reader.download_network("test", reader.sha256(b"verified"), temp)
            self.assertFalse((Path(temp) / reader.REVISION / "test.npz").exists())

    def test_hidden_permutation_preserves_actor(self):
        rng = np.random.default_rng(77)
        arrays = {"actor/dense1/kernel": rng.normal(size=(12, 5)),
                  "actor/dense1/bias": rng.normal(size=5),
                  "actor/dense2/kernel": rng.normal(size=(5, 6)),
                  "actor/dense2/bias": rng.normal(size=6),
                  "actor/logits/kernel": rng.normal(size=(6, 3)),
                  "actor/logits/bias": rng.normal(size=3)}
        result = reader.permutation_check(arrays, (2, 2, 3))
        self.assertTrue(result["passed"])
        self.assertEqual(result["binary_input_vectors"], 64)


if __name__ == "__main__":
    unittest.main()
