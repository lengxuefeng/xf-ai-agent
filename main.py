# -*- coding: utf-8 -*-
import os
import sys
import warnings
import uvicorn

# 统一将 app 目录加入导入路径，支持 `from services...` 这类无前缀导入风格
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app.main import app

# 过滤掉 langchain-tavily 的 pydantic 警告，不影响功能
warnings.filterwarnings("ignore", category=UserWarning, message=".*shadows an attribute in parent.*")

# 启动命令
# nohup python3 main.py > xf-ai-agent.log 2>&1 &
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
