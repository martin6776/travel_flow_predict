from settings import *
from GetDate import *
from FlowPredictTorchVersion import *
from FlowPredictKerasVersion import *

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
        sm1 = SceneModel('100',dt_start,dt_end,dt_predict)
        sm1.start_pipeline()
        end = time.time()
        print("耗时： ", end - start)
        break
