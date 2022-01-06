from settings import *
from MysqlConnectionTools import *
from NetworkModelKerasVersion import *
from DataPreprocessor import *
from InsertHistoryData import *
from GetDate import *

class SceneModelKeras:
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

    def data_preprocess(self):
        self.processor = DataPreprocessor(self.conn, self.dt_start, self.dt_end, self.dt_predict,
                                          self.scene_id)
        self.processor.preprocess_data()
        self.dataset_train, self.dataset_val, self.dataset_real_train, self.x_test, self.time_label = \
            self.processor.generate_train_val_test(lag,batch_size)

        columns = np.array(self.processor.preprocessed_df.columns.to_list())
        self.passenger_flow_index = np.where(columns=='fAll_count')[0]

    def save_model(self):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

    def feature_importance(self):
        feature_names = self.processor.preprocessed_df.columns.to_list()
        ig = IntegratedGradients(self.model)
        for inputs, target in self.dataset_val.take(1):
            ...
        inputs = torch.from_numpy(inputs.numpy()).to(torch.float32)
        inputs.requires_grad_()
        attr, delta = ig.attribute(inputs,target=0, return_convergence_delta=True)
        attr = attr.detach().numpy()
        self.visualize_importances(feature_names, np.mean(np.mean(attr, axis=0),axis=0)*100)

    # Helper method to print importances and visualize distribution
    def visualize_importances(self, feature_names, importances, title="Average Feature "
                                                     "Importances", plot=True, axis_title="Features"):
        plt.figure()
        print(title)
        for i in range(len(feature_names)):
            print(feature_names[i], ": ", '%.3f'%(importances[i]))
        x_pos = (np.arange(len(feature_names)))
        if plot:
            plt.figure(figsize=(12,6))
            plt.barh(x_pos, importances, align='center')
            plt.yticks(x_pos, feature_names, wrap=True)
            plt.ylabel(axis_title)
            plt.title(title)
            plt.savefig("evaluation/feature_importance.png")

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        new_lr = lr * (0.9 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def train(self):
        ''''''
        for inputs, target in self.dataset_train.take(1):
            ...
        self.model = LSTMModel(inputs.shape,hidden_dim)
        optimizer = keras.optimizers.Adam(learning_rate = lr,beta_1=0.9, beta_2=0.999)
        self.model.compile(loss = 'mae', optimizer = optimizer)
        history = self.model.fit(self.dataset_train, epochs = 300, verbose = 0,
                                 validation_data=self.dataset_val)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()
        plt.savefig("evaluation/loss_curve.png")

    def evaluate(self):
        """
        对模型进行评估，返回模型的rmse值
        :return:
        """
        y_hat_list = []
        y_true_list = []
        for batch in self.dataset_val:
            x_val, y_val = batch
            for i in range(x_val.shape[0]):
                y_hat_list.append(np.squeeze(self.model.predict(x_val[i:i+1,:,:])))
                y_true_list.append(y_val[i])
        y_hat_list = self.processor.scaler['fAll_count'].inverse_transform(np.array(y_hat_list).reshape(-1, 1))
        y_true_list = self.processor.scaler['fAll_count'].inverse_transform(np.array(y_true_list).reshape(-1, 1))
        plt.plot(y_hat_list, label='predict')
        plt.plot(y_true_list, label='true')
        plt.legend()
        plt.savefig("evaluation/eval_result_keras.png")

        self.rmse = np.sqrt(mean_squared_error(y_true_list, y_hat_list))
        self.mae = mean_absolute_error(y_true_list, y_hat_list)
        self.r2 = r2_score(y_true_list, y_hat_list)

        return self.rmse,self.mae,self.r2

    def predict(self):
        y_hat_list = []
        x_label = []
        current_date = datetime.datetime.now()
        for i in range(predict_day+1):
            input = self.x_test[i:(i+lag),:].reshape(1, lag, self.x_test.shape[1])
            y_hat = np.squeeze(self.model.predict(input))
            self.x_test[i+lag, self.passenger_flow_index] = y_hat
            y_hat_list.append(y_hat)
            x_label.append(self.time_label[i+lag].strftime('%Y-%m-%d'))


        y_hat = self.processor.scaler['fAll_count'].inverse_transform(np.array(y_hat_list) \
                                                                      .reshape(-1, 1))
        x_label = np.array(x_label)
        self.write_result(x_label, y_hat, current_date)

        return x_label, y_hat

    def write_result(self, dt, y_hat, f_date):
        for i in range(dt.shape[0]):
            self.conn.write_to_mysql_res2(self.scene_id, dt[i], y_hat[i], f_date, self.rmse,
                                          self.mae,
                                          self.r2)

if __name__ == '__main__':
    start = time.time()
    dt_start,dt_end,dt_predict = get_date()
    mysql = MysqlConnectTools(**MYSQL_CONFIG)
    current_date = datetime.datetime.now()
    delta = datetime.timedelta(days=-1)
    yesterday = (current_date + delta).strftime('%Y%m%d')
    # get scene id
    scene_list = mysql.get_sce_name(yesterday)
    print(scene_list)

    for i, scene_name in enumerate(scene_list):
        sm1 = SceneModelKeras('100',dt_start,dt_end,dt_predict)
        sm1.data_preprocess()
        sm1.train()
        rmse, mae, r2 = sm1.evaluate()
        print(f"模型结果评价: \n rmse: {rmse}, mae: {mae}, r2 score: {r2}")
        x, y = sm1.predict()
        for (date, predict) in zip(x, y):
            print(f"预测日期：{date}, 预测结果： {predict[0]}")


        end = time.time()
        print("耗时： ", end - start)
        break