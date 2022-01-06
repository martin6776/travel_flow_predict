"""
GetHoliday
节假日信息
"""
import numpy as np
import pandas as pd
from workalendar.asia import China

class GetHoliday:
    def __init__(self):
        pass

    def cal_festival(self, year_list = [2020,2021,2022]):
        cal = China()
        date_list = []
        for year in year_list:
            for x, v in cal.holidays(year):
                date_list.append([str(x), 1])
        df = pd.DataFrame(data=date_list, columns=['date', 'festival'])
        return df

    # 把时间列标准化时间格式
    def date_to_week(self, start_time, end_time):
        df = pd.DataFrame()
        df['date'] = pd.date_range(start=start_time, end=end_time)
        # 1-5表示工作日，6-7表示周末
        df['day_of_week'] = df['date'].dt.dayofweek + 1
        df['date'] = df['date'].map(lambda x: x.strftime('%Y-%m-%d'))
        return df

    def get_holiday_df(self, dt_start, dt_end):
        df_festival = self.cal_festival()
        df_date = self.date_to_week(dt_start, dt_end)
        df_filter_date = pd.merge(df_date, df_festival, on=["date"], how="left")
        df_filter_date["WORKING_DAY"] = np.select([(df_filter_date["day_of_week"]>5),(df_filter_date["day_of_week"]<6)],[1, 0])
        df_filter_date["OFFICIAL_HOLIDAY"] = np.select([(df_filter_date["day_of_week"]>5),(df_filter_date["day_of_week"]<6)],[0, 1])
        df_filter_date["LEGAL_HOLIDAY"] = np.select([(df_filter_date["festival"]==1),(df_filter_date["festival"]!=1)],[1, 0])
        df_filter_date= df_filter_date[["date", "LEGAL_HOLIDAY", "OFFICIAL_HOLIDAY", "WORKING_DAY"]]
        df_filter_date = df_filter_date.rename(columns = {'date':'t_date'})
        return df_filter_date

if __name__ == '__main__':
    holiday_info = GetHoliday()
    print(holiday_info.get_holiday_df('2021-12-01', '2021-12-07'))
    # print(holiday_info.cal_festival([2021]))
    # print(holiday_info.date_to_week('2021-12-01', '2021-12-07'))