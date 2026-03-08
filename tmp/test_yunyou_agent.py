import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from agent.graph_runner import GraphRunner
from utils.config import settings

def test_yunyou_agent():
    # Provide a simple model config
    model_config = {
        "model": "glm-4.7",
        "model_key": settings.API_KEY,
        "model_service": "local",
        "model_base_url": settings.BASE_URL,
        "temperature": 0.5
    }
    
    runner = GraphRunner(model_config=model_config)
    session_id = "test_yunyou_empty_result"
    
    print("Testing GraphRunner with '今天张三做了holter吗'...")
    generator = runner.stream_run("今天张三做了holter吗", session_id)
    
    for chunk in generator:
        print(chunk.strip())

if __name__ == "__main__":
    test_yunyou_agent()
