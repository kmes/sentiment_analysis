import socket
import os

POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

def test_db_up():
    print(f"Testing DB container connection on localhost:{POSTGRES_PORT}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', POSTGRES_PORT))
        sock.close()
        
        assert result == 0, f"Database port {POSTGRES_PORT} is not responsive. Container might be down."
        print(f"OK: DB container is up and listening on port {POSTGRES_PORT}.")
    except Exception as e:
        print(f"FAIL: DB status test failed: {e}")
        raise e

if __name__ == "__main__":
    test_db_up()
