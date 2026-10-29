import tempfile
import unittest
from pathlib import Path

from ruler_repro.data.manifest import latest_by_key, read_jsonl, write_jsonl


class ManifestTest(unittest.TestCase):
    def test_round_trip_and_latest_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = [{"sample_id": "a", "status": "error"}, {"sample_id": "a", "status": "success"}]
            write_jsonl(path, rows)
            loaded = read_jsonl(path)
            self.assertEqual(latest_by_key(loaded, "sample_id")["a"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
