import unittest
import json
import app
from app import app as flask_app

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = flask_app.test_client()
        self.app.testing = True

    def test_set_uid_valid(self):
        payload = {"uid": "24MCA20002"}
        response = self.app.post("/set_uid", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["ok"])
        self.assertIn("response", data)
        self.assertIn("suggestions", data)

    def test_set_uid_not_found(self):
        payload = {"uid": "INVALID_UID"}
        response = self.app.post("/set_uid", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertFalse(data["ok"])

    def test_set_uid_no_uid(self):
        payload = {}
        response = self.app.post("/set_uid", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 404)

if __name__ == "__main__":
    unittest.main()
