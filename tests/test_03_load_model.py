import urllib.request
import json
import os
import time

API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"

def test_load_and_check():
    try:
        print(f"Calling endpoint to load model: {BASE_URL}/load-model ...")
        load_req = urllib.request.Request(f"{BASE_URL}/load-model")
        with urllib.request.urlopen(load_req, timeout=30) as response:
            assert response.status == 200, f"Expected load-model status 200, got {response.status}"
            data = json.loads(response.read().decode())
            print(f"Load model response: {data}")
        
        # Adding a small delay to ensure the background task and state are fully settled, though it should be synchronous according to the route.
        time.sleep(1)
            
        print(f"Calling endpoint to check status: {BASE_URL}/status ...")
        status_req = urllib.request.Request(f"{BASE_URL}/status")
        with urllib.request.urlopen(status_req, timeout=5) as response:
            assert response.status == 200, f"Expected status 200, got {response.status}"
            data = json.loads(response.read().decode())
            print(f"Status response: {data}")
            assert data.get("model") == "loaded", "Model state should be 'loaded'"
            
        print("OK: Model was loaded correctly and status confirmed it.")
    except Exception as e:
        print(f"FAIL: Load model test failed: {e}")
        raise e

if __name__ == "__main__":
    test_load_and_check()
