import datetime
import settings
from MysqlConnectionTools import MysqlConnectTools
from GetHoliday import *
from GetWeather import *
from GetPatient import *

class InsertHistoryData:
    def __init__(self):
        self.current_date = datetime.datetime.now()
        self.delta = datetime.timedelta(days=-1)
        self.yesterday = (self.current_date + self.delta).strftime('%Y-%m-%d')

    def insert_holiday(self, conn: MysqlConnectTools):
        print("正在插入节假日数据")
        holiday_df = GetHoliday().get_holiday_df(dt_start='2020-01-01', dt_end='2022-12-31')
        t_date = holiday_df['t_date']
        LEGAL_HOLIDAY = holiday_df['LEGAL_HOLIDAY']
        OFFICIAL_HOLIDAY = holiday_df['OFFICIAL_HOLIDAY']
        WORKING_DAY = holiday_df['WORKING_DAY']
        for i in range(len(holiday_df)):
            conn.write_to_mysql_holiday_his(t_date[i], LEGAL_HOLIDAY[i], OFFICIAL_HOLIDAY[i], WORKING_DAY[i])

    def insert_weather(self, conn: MysqlConnectTools, start_date):
        print("正在插入天气数据")
        weather_df = GetWeather().get_history_weather_df(start_date)
        print(weather_df)
        f_date = weather_df['f_date']
        max_tempreture = weather_df['max_tempreture']
        min_tempreture = weather_df['min_tempreture']
        weather = weather_df['weather']
        for i in range(len(weather_df)):
            conn.write_to_mysql_weather_his(f_date.iloc[i], max_tempreture.iloc[i], \
                                            min_tempreture.iloc[i], weather.iloc[i])

    def insert_patient(self, conn: MysqlConnectTools, start_date):
        print("正在插入病患数据")
        patient_df = GetPatient().get_patient_df(dt_start=start_date)
        f_date = patient_df['f_date']
        currentConfirmedCount = patient_df['currentConfirmedCount']
        print(f_date, currentConfirmedCount)
        for i in range(len(patient_df)):
            conn.write_to_mysql_patient_his(str(f_date.iloc[i])[:10], currentConfirmedCount.iloc[i])



if __name__=='__main__':
    init = InsertHistoryData()
    mysql_conn = MysqlConnectTools(**settings.MYSQL_CONFIG)
    weather_curr_day, patient_curr_day = mysql_conn.get_most_current_day()
    print(weather_curr_day, patient_curr_day)
    init.insert_patient(mysql_conn)
