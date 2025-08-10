class AgentUtils:

    @staticmethod
    def to_sse(data) -> str:
        """
        将字符串或字典包装成符合 SSE (Server-Sent Events) 的格式。
        Args:
            data (str|dict): 需要发送的数据内容
        Returns:
            str: 符合 SSE 的数据格式，例如 'data: {"type": "thinking", "content": "..."}\n\n'
        """
        import json
        if isinstance(data, dict):
            data_str = json.dumps(data, ensure_ascii=False)
        else:
            data_str = str(data)
        return f"data: {data_str}\n\n"
