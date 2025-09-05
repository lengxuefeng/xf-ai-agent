import re
from datetime import datetime, timedelta


class DateUtils:
    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def parse_date_input(user_input: str) -> (str, str):
        today = datetime.today()

        if "昨天" in user_input:
            d = today - timedelta(days=1)
            return d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")

        if "今天" in user_input:
            return today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if "最近一周" in user_input or "过去一周" in user_input:
            start = today - timedelta(days=7)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        if "最近一个月" in user_input or "过去一个月" in user_input:
            start = today - timedelta(days=30)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

        match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
        if match:
            d = match.group(0)
            return d, d

        match_cn = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", user_input)
        if match_cn:
            d = datetime(int(match_cn.group(1)), int(match_cn.group(2)), int(match_cn.group(3)))
            return d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")

        return today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
