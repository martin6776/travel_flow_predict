from settings import *
from GetWeather import *


import settings
from MysqlConnectionTools import *

class DataPreprocessor:
    def __init__(self, mysql_con, dt_start, dt_end, dt_predict, scene_name='100'):
        self.scene_name = scene_name
        # 读取数据
        self.holiday_df = mysql_con.get_holiday(dt_start, dt_predict)
        self.weather_his_df = mysql_con.get_weather(dt_start, dt_end)
        # self.weather_his_df = mysql_con.get_weather(dt_start, dt_end)
        self.future_weather = GetWeather().scrap_future_weather()
        self.weather_df = pd.concat([self.weather_his_df, self.future_weather])
        self.weather_df[['max_tempreture','min_tempreture']] = self.weather_df[['max_tempreture',
                                                                                'min_tempreture']].astype('float64')
        self.patients_df = mysql_con.get_patients_count(dt_start, dt_end)
        self.patients_df['currentConfirmedCount'] = self.patients_df[
            'currentConfirmedCount'].astype('float64')

        self.passengers_df = mysql_con.get_yesterday_passenger_flow(self.scene_name, dt_start, dt_end).drop(columns = 'area_id')
        self.weibo_df = mysql_con.get_weibo(self.scene_name, dt_start, dt_end).drop(columns = 'SCE_NAME')
        self.wechat_df = mysql_con.get_wechat(self.scene_name, dt_start, dt_end).drop(columns = 'SCE_NAME')
        self.OTA_df = mysql_con.get_OTA_reviews_nums(self.scene_name, dt_start, dt_end).drop(columns = 'SCE_NAME')

        self.dt_start = dt_start
        self.dt_predict = dt_predict

    def preprocess_data(self):
        # 将读取的各类数据进行合并
        processed_df = pd.merge(self.holiday_df,self.OTA_df,how='outer',left_index=True,right_index=True)
        processed_df = pd.merge(processed_df,self.wechat_df,how='outer',left_index=True,right_index=True)
        processed_df = pd.merge(processed_df,self.weibo_df,how='outer',left_index=True,right_index=True)
        processed_df = pd.DataFrame(processed_df,dtype=np.float)
        for column in list(processed_df.columns[processed_df.isnull().sum() > 0]):
            mean_val = processed_df[column].mean()
            processed_df[column].fillna(mean_val, inplace=True)

        processed_df = pd.merge(processed_df,self.weather_df,how='outer',left_index=True,right_index=True)
        processed_df = pd.merge(processed_df,self.patients_df,how='outer',left_index=True,
                                right_index=True)
        processed_df = pd.merge(processed_df,self.passengers_df,how='outer',left_index=True,right_index=True)
        processed_df = processed_df.fillna(method='ffill')
        processed_df = processed_df.fillna(method='bfill')
        self.preprocessed_df = pd.get_dummies(processed_df)

        float_cols = self.preprocessed_df.select_dtypes('float').columns.to_list()
        # 使用字典保存scaler, 预测时直接使用字段名进行反归一化
        self.scaler = {}
        float_cols = float_cols[3:]
        for i, col_name in enumerate(float_cols):
            self.scaler[col_name] = MinMaxScaler(feature_range = (0, 1))
            self.preprocessed_df[col_name] = self.scaler[col_name].fit_transform(self.preprocessed_df[col_name].values.reshape(-1,1))
        print(self.preprocessed_df)
        return self.preprocessed_df

    def generate_train_val_test(self,lag, batch_size = 16):
        '''
        生成训练集和验证集用于验证模型损失函数，
        同时将全部数据构造一个新的训练集用于训练完整的模型用于预测
        param lag：滞后阶数，使用过去lag天的数据对未来进行预测
        param batch_size: 批训练每批数量
        '''
        self.preprocessed_df = self.preprocessed_df.loc[self.dt_start:self.dt_predict]
        x_train = self.preprocessed_df.iloc[:int(history_day*0.8)].values
        y_train = self.preprocessed_df['fAll_count'].iloc[lag:int(history_day*0.8) + lag].values

        x_val = self.preprocessed_df.iloc[int(history_day*0.8):-(lag + predict_day)].values
        y_val = self.preprocessed_df['fAll_count'].iloc[int(history_day*0.8) + lag:-predict_day].values


        x_test = self.preprocessed_df.iloc[-(lag + predict_day)-1:].values
        time_label = self.preprocessed_df.index[-(lag + predict_day)-1:]

        x_real_train = self.preprocessed_df.iloc[:-(lag + predict_day)].values
        y_real_train = self.preprocessed_df['fAll_count'].iloc[lag:-predict_day].values

        # 构造训练数据
        # 形成[[1,7,...720],[2,8,...,721]] 的x数据集，[[721],[722]...]的y数据集
        # 类型于滑动窗口取数一样
        dataset_train = keras.preprocessing.timeseries_dataset_from_array(
            x_train,
            y_train,
            sequence_length=lag, # 一批采集7天数据
            sampling_rate=1, # 1天采集一
            batch_size = batch_size
        )

        dataset_val = keras.preprocessing.timeseries_dataset_from_array(
            x_val,
            y_val,
            sequence_length=lag, # 一批采集7天数据
            sampling_rate=1, # 1天采集一
            batch_size = batch_size
        )

        dataset_real_train = keras.preprocessing.timeseries_dataset_from_array(
            x_real_train,
            y_real_train,
            sequence_length=lag, # 一批采集7天数据
            sampling_rate=1, # 1天采集一
            batch_size = batch_size
        )
        return dataset_train, dataset_val, dataset_real_train, x_test, time_label



if __name__ == '__main__':
    current_date = datetime.datetime.now()
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


    print(f"使用了从 {dt_start} 到 {dt_end} 的数据来训练模型。")
    print(f"对 {dt_end} 到 {dt_predict} 之间的数据进行了预测。")
    mysql = MysqlConnectTools(**settings.MYSQL_CONFIG)
    processor = DataPreprocessor(mysql, dt_start, dt_end, dt_predict, '100')
    # print(processor.OTA_df)
    print(processor.preprocess_data()['bad_num'])