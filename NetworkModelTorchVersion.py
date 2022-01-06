from settings import *

class TorchLSTM(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_layers):
        super(TorchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # input dimension (batch_sizze, lag, features_num)
        self.lstm = nn.LSTM(input_shape[2], hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim*lag, 1)
        self.sigmoid = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_out,_ = self.lstm(x)
        lstm_out = lstm_out.contiguous().view(batch_size, hidden_dim*lag)
        out = self.fc(lstm_out)
        return out