from settings import *
from NetworkModelTorchVersion import *

class ModelManager:
    def __init__(self, model):
        self.model = model

    def save_model(self):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))

    def feature_importance(self, processor, train, visualize = True):
        feature_names = processor.preprocessed_df.columns.to_list()
        attr_sum = np.zeros((len(feature_names)))
        ig = IntegratedGradients(self.model)

        for i, (inputs, target) in enumerate(train):
            inputs = torch.from_numpy(inputs.numpy()).to(torch.float32)
            inputs.requires_grad_()
            attr, delta = ig.attribute(inputs,target=0, return_convergence_delta=True)
            attr = attr.detach().numpy()
            attr = np.mean(np.mean(attr, axis=0),axis=0)
            attr_sum += attr
        if visualize:
            self.visualize_importances(feature_names, attr_sum/(i+1))
        return attr_sum/(i+1)

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

    def train(self, train):
        ''''''

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

        for i in range(epochs):
            # h = self.model.init_hidden(batch_size)
            new_lr = self.adjust_learning_rate(optimizer, i)
            for inputs, labels in train:
                inputs = torch.from_numpy(inputs.numpy()).to(torch.float32)
                labels = torch.from_numpy(labels.numpy()).to(torch.float32)
                self.model.zero_grad()
                output = self.model(inputs)
                # h = tuple([each.data for each in hidden])
                loss = criterion(output.squeeze(), labels)
                loss.backward(retain_graph=True)
                optimizer.step()

    def evaluate(self, validation, processor):
        """
        对模型进行评估，返回模型的rmse值
        :return:
        """
        self.model.eval()
        y_hat_list = []
        y_true_list = []
        for train, labels in validation:
            inputs = torch.from_numpy(train.numpy()).to(torch.float32)
            for i in range(inputs.shape[0]):
                y_hat = self.model(inputs[i:i+1,:,:])
                y_hat_list.append(np.squeeze(y_hat))
                y_true_list.append(labels[i])
        y_hat_list = processor.scaler['fAll_count'].inverse_transform(np.array(y_hat_list).reshape(-1, 1))
        y_true_list = processor.scaler['fAll_count'].inverse_transform(np.array(y_true_list).reshape(-1, 1))
        plt.plot(y_hat_list, label='predict')
        plt.plot(y_true_list, label='true')
        plt.legend()
        plt.savefig("evaluation/eval_result_torch.png")

        rmse = np.sqrt(mean_squared_error(y_true_list, y_hat_list))
        mae = mean_absolute_error(y_true_list, y_hat_list)
        r2 = r2_score(y_true_list, y_hat_list)

        return rmse,mae,r2

    def predict(self, test, label, index, processor):
        y_hat_list = []
        x_label = []
        current_date = datetime.datetime.now()
        for i in range(predict_day+1):
            input = test[i:(i+lag),:].reshape(1, lag, test.shape[1])
            input = torch.from_numpy(input).to(torch.float32)
            y_hat = np.squeeze(self.model(input).detach().numpy())
            test[i+lag, index] = y_hat
            y_hat_list.append(y_hat)
            x_label.append(label[i+lag].strftime('%Y-%m-%d'))


        y_hat = processor.scaler['fAll_count'].inverse_transform(np.array(y_hat_list) \
                                                                      .reshape(-1, 1))
        x_label = np.array(x_label)

        return x_label, y_hat, current_date

