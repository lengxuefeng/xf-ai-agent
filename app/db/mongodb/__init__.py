from pymongo import MongoClient

MONGO_URL = "mongodb://root:xiaoleng@127.0.0.1:27017/xf-ai-agent"

# 创建MongoDB数据库引擎
# 连接到 MongoDB 服务
client = MongoClient(MONGO_URL,
                     maxPoolSize=50,  # 最大连接池大小
                     minPoolSize=10,  # 最小连接池大小
                     socketTimeoutMS=60000,  # 60秒的socket超时
                     maxIdleTimeMS=60000,  # 60秒的最大空闲时间
                     heartbeatFrequencyMS=20000  # 20秒的心跳间隔
                     )

# 选择数据库（如果数据库不存在，会自动创建）
db = client['xf-ai-agent']

# 用户问答信息（如果表不存在，会自动创建）
chat_history = db['chat_history']

# 基础查询索引
db.chat_history.create_index([("session_id", 1)])  # 按会话查询
db.chat_history.create_index([("user_id", 1)])  # 按用户查询
db.chat_history.create_index([("created_at", -1)])  # 时间倒序

# 复合索引（常用查询场景）
db.chat_history.create_index([("user_id", 1), ("created_at", -1)])
