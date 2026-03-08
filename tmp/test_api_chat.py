import requests
import json

base_url = "http://127.0.0.1:8000"

def test_chat():
    # 1. Login
    login_data = {"username": "xuefeng", "password": "A715741925."}
    resp = requests.post(f"{base_url}/api/v1/auth/login", data=login_data)
    if resp.status_code != 200:
        print(f"Login failed: {resp.status_code} {resp.text}")
        return
        
    token = resp.json().get("access_token")
    
    # 2. Chat
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"message": "今天有哪些病人做了holter检查", "session_id": "test_empty_result"}
    
    chat_resp = requests.post(f"{base_url}/api/v1/chat/stream", headers=headers, json=payload, stream=True)
    
    for line in chat_resp.iter_lines():
        if line:
            print(line.decode("utf-8"))

if __name__ == "__main__":
    test_chat()
