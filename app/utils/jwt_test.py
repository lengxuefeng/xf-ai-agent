import secrets
import base64


def generate_hs256_key() -> str:
    """
    生成HS256算法所需的对称密钥（256位=32字节）
    返回Base64编码的密钥（便于存储和传输）
    """
    # 生成32字节的加密安全随机数
    key_bytes = secrets.token_bytes(32)
    # 编码为Base64字符串（去掉换行符）
    return base64.b64encode(key_bytes).decode().strip()


def generate_hs512_key() -> str:
    """生成HS512算法所需的对称密钥（512位=64字节）"""
    key_bytes = secrets.token_bytes(64)
    return base64.b64encode(key_bytes).decode().strip()


# 使用示例
if __name__ == "__main__":
    hs256_key = generate_hs256_key()
    hs512_key = generate_hs512_key()
    print(f"HS256密钥（32字节）: {hs256_key}")
    print(f"HS512密钥（64字节）: {hs512_key}")
