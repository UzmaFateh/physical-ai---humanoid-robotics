import requests
import json

# Test the backend API
def test_backend_connection():
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print("[OK] Backend is accessible")
            print(f"Health check response: {response.json()}")
        else:
            print(f"[ERROR] Backend health check failed with status {response.status_code}")
            return False

        # Test API v1 health endpoint
        response = requests.get('http://localhost:8000/api/v1/rag/health')
        if response.status_code == 200:
            print("[OK] API endpoint is accessible")
            print(f"API health response: {response.json()}")
        else:
            print(f"[ERROR] API endpoint failed with status {response.status_code}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Error connecting to backend: {e}")
        return False

if __name__ == "__main__":
    print("Testing backend connection...")
    success = test_backend_connection()
    if success:
        print("\n[OK] All tests passed! Backend is running and accessible.")
        print("The RAG chatbot should be able to connect to the backend at http://localhost:8000")
    else:
        print("\n[ERROR] Tests failed. Please check that the backend is running.")