import uvicorn

from app.main import app

# 启动命令
# nohup python3 main.py > xf-ai-agent.log 2>&1 &
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
