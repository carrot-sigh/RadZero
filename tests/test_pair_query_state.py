import tempfile
import unittest
from pathlib import Path

import torch

from exp.cxr_pt.pair_query_state import (
    anatomy_state_diagnostics,
    load_static_anatomy_state,
    save_static_anatomy_state,
)


class PairQueryStateTest(unittest.TestCase):
    def test_cache_round_trip_and_provenance_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            state = torch.randn(3, 8)
            save_static_anatomy_state(path, state, {"indices_sha256": "abc", "sample_count": 5})
            loaded, metadata = load_static_anatomy_state(
                path, {"indices_sha256": "abc", "sample_count": 5}
            )
            torch.testing.assert_close(loaded, state)
            self.assertEqual(metadata["shape"], [3, 8])
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                load_static_anatomy_state(path, {"indices_sha256": "wrong"})

    def test_diagnostics_are_finite(self):
        states = torch.randn(5, 3, 8)
        result = anatomy_state_diagnostics(states, states.mean(dim=0), [(0, 1)])
        self.assertEqual(result["sample_count"], 5)
        self.assertEqual(result["left_right"][0]["count"], 5)


if __name__ == "__main__":
    unittest.main()
