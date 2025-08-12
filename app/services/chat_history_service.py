from datetime import datetime

from db.mongodb import db


class ChatHistoryService:
    """
        MongoDB 自动生成的 _id 里面包含了时间信息。
       _id 是 MongoDB 中用于标识文档的唯一字段。它是一个 12 字节的 ObjectID，其生成规则如下：
       4 字节的时间戳：表示生成 _id 的时间，精确到秒级别。
       3 字节的机器标识：表示生成该 _id 的机器标识。
       2 字节的进程 id：表示生成该 _id 的进程 id。
       3 字节的自增计数器：用于在同一秒内生成不同文档的计数器。
       """

    def __init__(self):
        self.chat_history = db['chat_history']

    def add_chat_history(self, chat: dict):
        """
        保存用户对话内容到mongo

        Args:
            chat: 对话信息

        Returns:
            chat: 对话信息
        """
        chat['is_deleted'] = 0
        chat['created_at'] = datetime.now()
        self.chat_history.insert_one(chat)
        return chat

    def get_chat_history(self, user_id: str):
        """
                获取用户对话列表

                Args:
                    user_id: 用户id

                Returns:
                    对话列表
                """
        return self.chat_history.find_one({"user_id": user_id})

    def get_chat_history_list(self, user_id: str):
        """
        获取用户对话列表

        Args:
            user_id: 用户id

        Returns:
            对话列表
        """
        return self.chat_history.find({"user_id": user_id}).sort("created_at", -1).skip(0).limit(10)

