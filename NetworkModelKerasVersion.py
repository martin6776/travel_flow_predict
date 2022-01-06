from settings import *

class LSTMModel(Model):
    def __init__(self, input_dim, hidden_dim = 20):
        super(LSTMModel, self).__init__()
        self.lstm1 = Bidirectional(LSTM(hidden_dim, input_shape=(input_dim[1], input_dim[2]),
                                       return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(hidden_dim, input_shape=(input_dim[1], input_dim[2])))
        self.fc = Dense(1, activation='softsign')

    def call(self, x):
        y = self.lstm2(self.lstm1(x))
        return self.fc(y)
