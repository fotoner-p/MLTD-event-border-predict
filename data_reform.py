import os
import numpy as np
import pandas as pd
import json
import datetime

idol_value: pd.DataFrame = pd.read_csv('./anniversary/all_idol_value.csv')
reform_value = idol_value.copy()

for idx in idol_value.columns:
    mean = idol_value[idx].mean()
    std = idol_value[idx].std()
    reform_value[idx] = (idol_value[idx] - mean) / std

reform_value.to_csv('./anniversary/reform_idol_value.csv', index=False)

data_type = "./tour/"
data_path = data_type + 'csv/'
save_path = data_type + 'reform_csv/'
csv_list = os.listdir(data_path)

data_list = []
len_list = []
for csv_name in csv_list:
    event_data: pd.DataFrame = pd.read_csv(data_path + csv_name)
    data_list.append(event_data)
    len_list.append(len(event_data))

len_mean = np.mean(len_list)
len_std = np.std(len_list)

sum_list = data_list[0].append(data_list[1:len(csv_list)])

mean_std_dict = {}

for idx in sum_list.columns:
    data_mean = sum_list[idx].mean()
    data_std = sum_list[idx].std()

    mean_std_dict[idx] = {
        'data_mean': data_mean,
        'data_std': data_std
    }

with open("./data_mean_std.json", "w", encoding='utf-8') as file:
    json.dump(mean_std_dict, file, ensure_ascii=False, indent="\t")

for i, data in enumerate(data_list):
    event_idx_str = csv_list[i][:-4]
    reform_event = data.copy()

    for idx in sum_list.columns:
        data_mean = sum_list[idx].mean()
        data_std = sum_list[idx].std()
        reform_event[idx] = (data[idx] - data_mean) / data_std

    reform_event.to_csv(save_path + '%s.csv' % event_idx_str, index=False)

with open(data_type + "event_list.json", "r", encoding="utf-8") as file:
    event_json = json.load(file)

run_time_list = []
boost_time_list = []

for event in event_json:
    begin = datetime.datetime.strptime(event['schedule']["beginDate"], "%Y-%m-%dT%H:%M:%S+09:00")
    boost = datetime.datetime.strptime(event['schedule']["boostBeginDate"], "%Y-%m-%dT%H:%M:%S+09:00")
    end = datetime.datetime.strptime(event['schedule']["endDate"], "%Y-%m-%dT%H:%M:%S+09:00")

    run_time = end - begin
    run_time_sec = run_time.days * 24 * 60 * 60 + run_time.seconds + 1

    boost_time = boost - begin
    boost_time_sec = boost_time.days * 24 * 60 * 60 + boost_time.seconds + 1

    run_time_list.append(run_time_sec // 60 // 30)
    boost_time_list.append(boost_time_sec // 60 // 30)

run_time_mean = np.mean(run_time_list)
run_time_std = np.std(run_time_list)

boost_time_mean = np.mean(boost_time_list)
boost_time_std = np.std(boost_time_list)

time_dict = {
    "run_time_mean": run_time_mean,
    "run_time_std": run_time_std,
    "boost_time_mean": boost_time_mean,
    "boost_time_std": boost_time_std
}

for i, event in enumerate(event_json):
    time_dict[str(event['id'])] = {
        "run_time_len": run_time_list[i],
        "boost_time_len": boost_time_list[i],
    }

with open("./run_time_len.json", "w", encoding='utf-8') as file:
    json.dump(time_dict, file, ensure_ascii=False, indent="\t")