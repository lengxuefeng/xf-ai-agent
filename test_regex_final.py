import re

texts = [
    "你是谁，昨天的时间是什么",
    "你是谁，现在的时间是什么",
    "今天几号了，现在的时间是什么，昨天的时间",
    "今天是几号",
    "明天的日期"
]

patterns = [
    r"^(?!.*(昨天|明天|前天|后天|昨儿|明儿)).*(今天几号|几号了|是什么日期|今天星期几|的日期)",
    r"^(?!.*(昨天|明天|前天|后天|昨儿|明儿)).*(现在几点|当前时间|现在的时间|几点了|什么时间|时间是什么)"
]

for t in texts:
    print(f"\nEvaluating: '{t}'")
    for p in patterns:
        if re.search(p, t):
            print(f"  MATCHES: {p}")
        else:
            print(f"  NO MATCH: {p}")
