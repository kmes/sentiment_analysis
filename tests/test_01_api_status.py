import urllib.request
import json
import os

API_PORT = os.getenv("API_PORT", "8000")
URL = f"http://localhost:{API_PORT}/status"

def test_api_up():
    print(f"Testing API status on {URL}...")
    try:
        req = urllib.request.Request(URL)
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200, f"Expected status 200, got {response.status}"
            data = json.loads(response.read().decode())
            assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
            print("OK: API is up and /status returns 'ok'.")
    except Exception as e:
        print(f"FAIL: API status test failed: {e}")
        raise e

if __name__ == "__main__":
    test_api_up()
