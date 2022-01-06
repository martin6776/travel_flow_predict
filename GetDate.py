from settings import *

def get_date():
    current_date = datetime.datetime.now()
    # current_date = datetime.datetime.strptime("2021-12-21","%Y-%m-%d")
    # 使用历史400天的数据训练模型
    day_start_disparity = datetime.timedelta(days=-history_day)
    # 预测未来7天数据
    day_predict_disparity = datetime.timedelta(days=predict_day)
    days_start = current_date + day_start_disparity
    days_predict = current_date + day_predict_disparity
    # 日期格式统一为 Y-m-d
    dt_start = days_start.strftime('%Y-%m-%d')
    dt_end = current_date.strftime('%Y-%m-%d')
    dt_predict = days_predict.strftime('%Y-%m-%d')

    return dt_start,dt_end,dt_predict