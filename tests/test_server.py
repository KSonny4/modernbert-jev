"""Server dispatch tests; need MJEV_TEST_MODEL pointing at a model dir."""

import json
import os
import unittest

from modernbert_jev.server import dispatch, find_models

MODEL_DIR = os.environ.get("MJEV_TEST_MODEL", "")


class DispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_DIR or not os.path.isfile(os.path.join(MODEL_DIR, "model.onnx")):
            raise unittest.SkipTest("set MJEV_TEST_MODEL to a model dir")
        from modernbert_jev.runtime import DecisionModel

        cls.model = DecisionModel(MODEL_DIR, threads=1)
        cls.models_dir = os.path.dirname(os.path.abspath(MODEL_DIR))

    def test_systemone_roundtrip(self):
        body = json.dumps(
            {
                "state": "The screen arrived cracked.",
                "model": "modernbert-jev",
                "questions": {
                    "damaged": {"type": "noul", "instructions": "The item arrived damaged"}
                },
            }
        ).encode()
        status, payload = dispatch(self.model, self.models_dir, "POST", "/v1/systemone", body)
        self.assertEqual(status, 200)
        self.assertGreaterEqual(payload["answers"]["damaged"]["noul"], 0.5)

    def test_models_list(self):
        status, payload = dispatch(self.model, self.models_dir, "GET", "/v1/models", b"")
        self.assertEqual(status, 200)
        names = [m["name"] for m in payload["models"]]
        self.assertIn(os.path.basename(MODEL_DIR.rstrip("/")), names)
        self.assertEqual(names, find_models(self.models_dir))

    def test_bad_request_is_400(self):
        status, _ = dispatch(
            self.model, self.models_dir, "POST", "/v1/systemone", b'{"state": "x"}'
        )
        self.assertEqual(status, 400)

    def test_unknown_path_is_404(self):
        status, _ = dispatch(self.model, self.models_dir, "GET", "/nope", b"")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
