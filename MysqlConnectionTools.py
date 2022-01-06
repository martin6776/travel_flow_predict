from settings import *
from sqlalchemy import create_engine
import pandas as pd
import pymysql

class MysqlConnectTools:
    def __init__(self, **kwargs):
        self.conn = pymysql.connect(**kwargs)
        self.host = MYSQL_CONFIG['host']
        self.port = MYSQL_CONFIG['port']
        self.username = MYSQL_CONFIG['user']
        self.password = MYSQL_CONFIG['passwd']
        self.db = MYSQL_CONFIG['db']

    def get_sce_name(self, yesterday):
        sql = f"select distinct area_id from yesterday_passenger_flow_xw WHERE f_date = '{yesterday}'"
        df = pd.read_sql(sql, self.conn)
        sce_list = df.values.tolist()
        return sce_list

    def get_holiday(self, dt_start, dt_end):
        sql = f"SELECT t_date, LEGAL_HOLIDAY, OFFICIAL_HOLIDAY, WORKING_DAY FROM {holiday_table} where " \
              f"t_date <= '{dt_end}' and t_date >= '{dt_start}'"
        df = pd.read_sql(sql, self.conn, index_col="t_date")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        if 'LEGAL_HOLIDAY' not in df.columns:
            df['LEGAL_HOLIDAY'] = 0
        return df

    def get_OTA_reviews_nums(self, sceId, dt_start, dt_end):
        sql = "select T_DATA, SCE_NAME, sum(cmt_num) cmt_num, sum(good_num)/sum(cmt_num) good_num, " \
              "    sum(middle_num)/sum(cmt_num) middle_num, sum(bad_num)/sum(cmt_num) bad_num " \
              "from " \
              "    (select DISTINCT T_DATA,SCE_NAME,cmt_num,good_num,middle_num,bad_num " \
              "    from ota_reviews_number_scenic_ranking " \
              "    where SCE_NAME='%s' and T_DATA<='%s' and T_DATA>='%s') a " \
              "group by T_DATA, SCE_NAME" % (sceId, dt_end, dt_start)
        df = pd.read_sql(sql, self.conn, index_col="T_DATA")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        if df.empty:
            cursor = self.conn.cursor()
            insql = "insert into ota_reviews_number_scenic_ranking values('%s','%s','0','0','0','0')" % (
                dt_end, sceId)
            cursor.execute(insql)
            self.conn.commit()
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df

    def get_wechat(self, sceId, dt_start, dt_end):
        sql = "select T_DATE,SCE_NAME,sum(WECHAT_NUM) WECHAT_NUM " \
              "from " \
              "    (select T_DATE,SCE_NAME,WECHAT_NUM " \
              "    from wechat_ound_volume_scenic_ranking " \
              "    where SCE_NAME='%s' and T_DATE<='%s' and T_DATE>='%s') a " \
              "group by T_DATE,SCE_NAME" % (sceId, dt_end, dt_start)
        df = pd.read_sql(sql, self.conn, index_col="T_DATE")
        if df.empty:
            cursor = self.conn.cursor()
            insql = """insert into wechat_ound_volume_scenic_ranking values('%s','%s','0')""" % (dt_end, sceId)
            cursor.execute(insql)
            self.conn.commit()
            df = pd.read_sql(sql, self.conn, index_col="T_DATE")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df

    def get_weibo(self, sceId, dt_start, dt_end):
        sql = "select T_DATE,SCE_NAME,sum(WEIBO_NUM) WEIBO_NUM " \
              "from " \
              "    (select T_DATE,SCE_NAME,WEIBO_NUM " \
              "    from weibo_ound_volume_scenic_ranking " \
              "    where SCE_NAME='%s' and T_DATE<='%s' and T_DATE >= '%s') a " \
              "group by T_DATE,SCE_NAME" % (sceId, dt_end, dt_start)
        df = pd.read_sql(sql, self.conn, index_col="T_DATE")
        if df.empty:
            cursor = self.conn.cursor()
            insql = """insert into weibo_ound_volume_scenic_ranking values('%s','%s','0')""" % (dt_end, sceId)
            cursor.execute(insql)
            self.conn.commit()
            df = pd.read_sql(sql, self.conn, index_col="T_DATE")
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        return df

    def get_yesterday_passenger_flow(self, sceId, dt_start, dt_end):
        dt1 = dt_start.replace("-", "")
        dt2 = dt_end.replace("-", "")
        sql = "select area_id, from_unixtime(unix_timestamp(f_date)) AS f_date,sum(fAll_count) fAll_count " \
              "from " \
              "    (select distinct area_id,f_date,fAll_count " \
              "    from yesterday_passenger_flow_xw " \
              "    where area_id=%s and f_date<='%s' and f_date >='%s') a " \
              "group by area_id,f_date" % (
                  sceId, dt2, dt1)

        df = pd.read_sql(sql, self.conn, index_col="f_date")
        return df

    ##########new code from here##################################################

    def get_weather(self, dt_start, dt_end):
        sql = f"SELECT DISTINCT * from {weather_table} where f_date<='{dt_end}' and f_date>='{dt_start}'"
        df = pd.read_sql(sql, self.conn, index_col = "f_date")
        return df

    def get_patients_count(self,dt_start, dt_end):
        sql = f"""SELECT DISTINCT f_date, currentConfirmedCount 
              FROM {patient_table} where f_date<='{dt_end}' and f_date>='{dt_start}'"""
        df = pd.read_sql(sql, self.conn, index_col="f_date")
        return df
    ############################################################

    def write_to_mysql_res2(self, sec_name, dt, SUM, f_date, RMSE, MAE, R2):
        cursor = self.conn.cursor()
        # insert into需要添加列信息
        # insert into table(col1, col2) values(...)
        # trend_of_scenic_spots_train_result
        sql1 = f"replace into {predict_table} (scenic_spots, dt, num, update_time, RMSE, MAE, R2)" \
               f" values ('{sec_name}','{dt}','{SUM}','{f_date}', '{RMSE}', '{MAE}', '{R2}')"
        cursor.execute(sql1)

        self.conn.commit()

    ########################################
    # 插入历史节假日数据
    def write_to_mysql_holiday_his(self, t_date, LEGAL_HOLIDAY, OFFICIAL_HOLIDAY, WORKING_DAY):
        cursor = self.conn.cursor()
        sql = f"replace into {holiday_table} (t_date, LEGAL_HOLIDAY, OFFICIAL_HOLIDAY, WORKING_DAY) " \
              f"values('{holiday_table}', '{t_date}', '{LEGAL_HOLIDAY}','{OFFICIAL_HOLIDAY}','{WORKING_DAY}')"
        cursor.execute(sql)
        self.conn.commit()

    # 插入历史天气数据
    def write_to_mysql_weather_his(self, f_date, max_tempreture, min_tempreture, weather):
        cursor = self.conn.cursor()
        # t_weather_info
        sql = f"""replace into {weather_table} (f_date, max_tempreture, min_tempreture, weather) 
                    values('{f_date}', '{max_tempreture}','{min_tempreture}','{weather}')"""
        cursor.execute(sql)
        self.conn.commit()

    # 插入历史确诊人数数据
    def write_to_mysql_patient_his(self, f_date, currentConfirmedCount):
        cursor = self.conn.cursor()
        sql = f"replace into {patient_table} (f_date, currentConfirmedCount) values('{f_date}'" \
              f",'{currentConfirmedCount}')"
        cursor.execute(sql)
        self.conn.commit()


    # 创建表
    def create_tables(self):
        cursor = self.conn.cursor()
        try:
            sql_creat1 = f"""CREATE TABLE if not exists {holiday_table}(
                        t_date VARCHAR(50),
                        LEGAL_HOLIDAY INT,
                        OFFICIAL_HOLIDAY INT,
                        WORKING_DAY INT
                        )"""
            cursor.execute(sql_creat1)
        except UserWarning:
            print("建表失败")

        try:
            sql_creat2 = f"""CREATE TABLE if not exists {patient_table}(
                        f_date VARCHAR(50),
                        currentConfirmedCount INT
                        )"""
            cursor.execute(sql_creat2)
        except UserWarning:
            print("建表失败")

        try:
            sql_creat3 = f"""CREATE TABLE if not exists {weather_table}(
                        f_date VARCHAR(50),
                        max_tempreture INT,
                        min_tempreture INT,
                        weather VARCHAR(50)
                        )"""
            cursor.execute(sql_creat3)
        except UserWarning:
            print("建表失败")

    # 写入昨天的天气和确诊人数
    def mysql_connect(self):
        host = self.host
        port = self.port
        username = self.username
        password =self.password
        db = self.db
        connect_url = 'mysql+pymysql://' + str(username) + ':' + str(password) + '@' + str(host) + ':' + str(port) + '/' + db + '?charset=utf8'
        mysql_connect = create_engine(connect_url)
        return mysql_connect

    def get_most_current_day(self, dt_start):
        sql1 = f"SELECT f_date from {weather_table} ORDER BY f_date DESC LIMIT 1"
        wea_df = pd.read_sql(sql1, self.conn)
        sql2 = f"SELECT f_date from {patient_table} ORDER BY f_date DESC LIMIT 1"
        patient_df = pd.read_sql(sql2, self.conn)
        sql3 = f"SELECT t_date from {holiday_table} ORDER BY t_date DESC LIMIT 1"
        holiday_df = pd.read_sql(sql3, self.conn)

        wea_list = wea_df['f_date'].values.tolist()
        patient_list = patient_df['f_date'].values.tolist()
        holiday_list = holiday_df['t_date'].values.tolist()

        if len(wea_list) == 0:
            wea_date = dt_start
        else:
            wea_date = wea_list[0]

        if len(patient_list) == 0:
            patient_date = dt_start
        else:
            patient_date = patient_list[0]

        if len(holiday_list) == 0:
            holiday_date = dt_start
        else:
            holiday_date = holiday_list[0]

        return (wea_date, patient_date, holiday_date)

if __name__=='__main__':
    mysql = MysqlConnectTools(**MYSQL_CONFIG)
    mysql.get_holiday("2021-12-01", "2021-12-08")
    # mysql.write_to_mysql_weather_day()
    # mysql.write_to_mysql_patient_day()
