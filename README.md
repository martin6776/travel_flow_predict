# travel_flow_predict

## 1. Data Input
- Used features: ['LEGAL_HOLIDAY', 'OFFICIAL_HOLIDAY', 'WORKING_DAY', 'cmt_num',
       'good_num', 'middle_num', 'bad_num', 'WECHAT_NUM', 'WEIBO_NUM',
       'max_tempreture', 'min_tempreture', 'currentConfirmedCount',
       'fAll_count_yesterday', 'weather_多云', 'weather_晴', 'weather_雨', 'weather_雪']

- predict label: future travel flow count

## 2. Data Preprocess



## 3. Model Structure
- TorchLSTM(


  (lstm): LSTM(17, 32, num_layers=2, batch_first=True)
  
  
  (fc): Linear(in_features=160, out_features=1, bias=True)
  
  
  (sigmoid): ReLU()
  
  
)


## 4. Result
![avatar](/evaluation/eval_result_torch.png)

![avatar](/evaluation/feature_importance.png)
