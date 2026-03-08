import re

texts = [
    "你是谁，昨天的时间是什么",
    "你是谁，现在的时间是什么",
    "明天的日期",
    "今天的日期是什么"
]

patterns = [
    r"^(?!.*(昨天|明天|前天|后天|明儿|昨儿)).*什么时间",
    r"^(?!.*(昨天|明天|前天|后天)).*时间是什么",
    r"^(?!.*(昨天|明天|前天|后天)).*是什么日期",
    r"^(?!.*(昨天|明天|前天|后天)).*的日期"
]

for t in texts:
    print(f"\nEvaluating: '{t}'")
    for p in patterns:
        if re.search(p, t):
            print(f"  MATCHES: {p}")
        else:
            print(f"  NO MATCH: {p}")
