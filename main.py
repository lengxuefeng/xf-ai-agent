# -*- coding: utf-8 -*-
import warnings
import uvicorn

from app.main import app

# 过滤掉 langchain-tavily 的 pydantic 警告，不影响功能
warnings.filterwarnings("ignore", category=UserWarning, message=".*shadows an attribute in parent.*")

# 启动命令
# nohup python3 main.py > xf-ai-agent.log 2>&1 &
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
