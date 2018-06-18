"""
呼び出されたタイミングの時刻を取得し，時刻に応じたメッセージを返します
"""

import datetime


def get_time():

    time = datetime.datetime.now()
    time_str = time.strftime("%Y/%m/%d %H:%M:%S")
    return time_str, time


def print_msg(now_time):

    start_coretime = now_time.replace(
        hour=11, minute=0, second=0, microsecond=0
    )
    start_day = now_time.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_coretime = now_time.replace(
        hour=15, minute=0, second=0, microsecond=0
    )
    if start_day <= now_time <= start_coretime:
        return "出席"
    elif start_coretime < now_time <= end_coretime:
        return "遅刻"
    else:
        return "欠席"


def main():

    time_str, now_time=get_time()
    msg = print_msg(now_time)
    print(time_str, msg)


if __name__ == '__main__':
    main()
