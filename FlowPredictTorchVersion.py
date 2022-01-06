from settings import *
from MysqlConnectionTools import *
from ModelManager import *
from DataPreprocessor import *
from InsertHistoryData import *

class SceneModel:
    def __init__(self, scene_id, dt_start, dt_end, dt_predict):
        self.scene_id = scene_id
        self.conn = MysqlConnectTools(**MYSQL_CONFIG)
        self.dt_start = dt_start
        self.dt_end = dt_end
        self.dt_predict = dt_predict

        self.current_date = datetime.datetime.today()
        self.delta = datetime.timedelta(days=-1)
        self.yesterday = datetime.datetime.strptime((self.current_date + self.delta).strftime('%Y-%m-%d'),"%Y-%m-%d")
        print(f"景区{self.scene_id}人流量预测")
        print(f"使用了从{self.dt_start}到{self.dt_end}的数据来训练模型。")
        print(f"对{self.dt_end}  到{self.dt_predict}之间的数据进行了预测。")
        self.check_history_data()
        self.conn.create_tables()

    def check_history_data(self):
        """更新历史天气，病患数据"""
        check_history = InsertHistoryData()
        weather_curr_day, patient_curr_day, holiday_curr_day= self.conn.get_most_current_day(self.dt_start)
        print(self.yesterday)
        print("天气表最新日期", weather_curr_day, "\n病患表最新日期",patient_curr_day, "\n节假日表最新日期", holiday_curr_day)

        if datetime.datetime.strptime(weather_curr_day,"%Y-%m-%d") < self.yesterday:
            check_history.insert_weather(self.conn, weather_curr_day)
        if datetime.datetime.strptime(patient_curr_day,"%Y-%m-%d") < self.yesterday:
            check_history.insert_patient(self.conn, patient_curr_day)
        if (holiday_curr_day) == 'isEmpty':
            check_history.insert_holiday(self.conn)

    def get_data_processor(self):
        """创建数据处理器"""
        processor = DataPreprocessor(self.conn, self.dt_start, self.dt_end, self.dt_predict,
                                          self.scene_id)
        processor.preprocess_data()
        columns = np.array(processor.preprocessed_df.columns.to_list())
        passenger_flow_index = np.where(columns == 'fAll_count')[0]
        return processor, passenger_flow_index

    def create_model(self, train):
        """构建模型，"""
        for inputs, target in train.take(1):
            ...
        model = TorchLSTM(inputs.shape,hidden_dim, layers)
        model_manager = ModelManager(model)

        return model_manager

    def write_result(self, dt, y_hat, f_date, rmse, mae, r2):
        for i in range(dt.shape[0]):
            self.conn.write_to_mysql_res2(self.scene_id, dt[i], int(y_hat[i][0]), f_date, rmse,
                                                                    mae,r2)

    def start_pipeline(self):
        # 获取数据处理器
        processor, passenger_flow_index = self.get_data_processor()
        # 分割数据集，返回第一个为模型评估训练集，第二个为验证集，第三个为预测结果训练集，第四个为预测数据，第五个为预测数据的时间标签
        _, dataset_val, dataset_real_train, x_test, time_label = \
                                processor.generate_train_val_test(lag ,batch_size)
        # 构建模型，导入预训练模型，训练模型，评估模型，加用模型预测实际数据
        model_manager = self.create_model(dataset_real_train)
        model_manager.load_model()

        model_manager.train(dataset_real_train)
        rmse,mae,r2 = model_manager.evaluate(dataset_val, processor)

        feature_importance = model_manager.feature_importance(processor,dataset_real_train)

        x_label, y_hat, current_date = model_manager.predict(x_test, time_label,
                                                           passenger_flow_index, processor)

        # 存储结果
        self.write_result(x_label, y_hat, current_date, rmse,mae,r2)
        model_manager.save_model()
        print(f"模型评价指标：rmse {rmse}, mae {mae}, r2 {r2}")
        for (date, predict) in zip(x_label, y_hat):
            print(f"预测日期：{date}, 预测结果： {predict[0]}")







