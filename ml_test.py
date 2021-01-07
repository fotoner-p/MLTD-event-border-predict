import json
from model import *
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from typing import List

extract_border: List[int] = [
    100,
    2500,
    5000,
    10000,
    25000,
    50000
]
seq_length = 144

def multivariate_data(dataset, target, window_size):
    data = []
    labels = []
    dataset = np.array(dataset)
    for i in range(window_size, len(dataset) - window_size):
        pre_indices = range(i - window_size, i)
        after_indices = range(i, i + window_size)
        data.append(dataset[pre_indices].tolist())

        labels.append(dataset[after_indices, target].tolist())
    # print(indices)
    # print(i)

    return data, labels


model = MLTD_model(data_dim=4, seq_length=seq_length)
model.load_weights('./2500_model.h5')

idol_value: pd.DataFrame = pd.read_csv('./anniversary/reform_idol_value.csv')

with open("./data_mean_std.json", "r", encoding='utf-8') as file:
    mean_std_dict = json.load(file)

with open("./run_time_len.json", "r", encoding='utf-8') as file:
    time_dict = json.load(file)

with open("./cards/event.json", "r", encoding='utf-8') as file:
    event_idol = json.load(file)

event_id = 156

extract_border_str = ",".join([str(border) for border in extract_border])

res = requests.get("https://api.matsurihi.me/mltd/v1/events/%d/rankings/logs/eventPoint/%s" % (event_id,
                                                                                               extract_border_str))

res_json = res.json()
rank_list: List[List[int]] = [[] for _ in range(len(extract_border))]

for i in range(len(res_json[0]['data'])):
    for j in range(len(extract_border)):
        try:
            rank_list[j].append(int(res_json[j]['data'][i]['score']))
        except IndexError:
            rank_list[j].insert(0, 0)

rank_result = {}

for i, cur_list in enumerate(rank_list):
    idx = '#' + str(extract_border[i])
    rank_result[idx] = (np.array(cur_list) - mean_std_dict[idx]["data_mean"]) / mean_std_dict[idx]["data_std"]

event_data = pd.DataFrame(rank_result)
rank_idol = event_idol[str(event_id)]['ranking'] - 1
rank_vector = idol_value.loc[rank_idol].values.tolist()

point_idol = event_idol[str(event_id)]['point'] - 1
point_vector = idol_value.loc[point_idol].values.tolist()

run_len = time_dict[str(event_id)]["run_time_len"]
boost_len = time_dict[str(event_id)]["boost_time_len"]

run_norm_len = (float(run_len) - time_dict["run_time_mean"]) / time_dict["run_time_std"]
boost_norm_len = (float(boost_len) - time_dict["boost_time_mean"]) / time_dict["boost_time_std"]

# point_data = []
#
# for i in range(len(event_data)):
#     try:
#         event_value = event_data.loc[i]["#2500"]
#         # event_value = event_data.loc[i].values.tolist()
#     except Exception as e:
#         print(i, run_len, len(event_data))
#         break
#     cur_len = float(i + 1) / run_len
#     # cur_data = [cur_len, run_norm_len, boost_norm_len] + event_value # + rank_vector  + point_vector
#     cur_data = [cur_len, run_norm_len, boost_norm_len, event_value]  # + rank_vector  + point_vector
#
#     point_data.append(cur_data)
#
# test_x, test_y = multivariate_data(point_data, 3, seq_length)
# result_y = model.predict(test_x)

# real = (np.array(test_y) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
# predict = (np.array(result_y) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
# plt.grid(True)
# plt.plot(real, label="real")
# plt.plot(predict, linestyle="--", label="predict")
# plt.legend(loc='lower right', frameon=False)
# plt.show()


cur_seq = []
max_idx = 0
for i in range(len(event_data) - seq_length, len(event_data)):
    max_idx = i
    try:
        event_value = event_data.loc[i]["#2500"]
        #event_value = event_data.loc[i].values.tolist()
    except Exception as e:
        print(i, run_len, len(event_data))
        break
    cur_len = float(i + 1) / run_len
    # cur_data = [cur_len, run_norm_len, boost_norm_len] + event_value # + rank_vector #+ point_vector
    cur_data = [cur_len, run_norm_len, boost_norm_len, event_value] # + event_value# + rank_vector #+ point_vector

    cur_seq.append(cur_data)

predict_list = []

for i in range(max_idx, run_len - seq_length, seq_length):
    predict_value = model.predict([cur_seq])
    predict_list += predict_value[0].tolist()

    cur_seq = []
    for j in range(0, seq_length):
        cur_len = float(j + i + 1) / run_len
        # cur_data = [cur_len, run_norm_len, boost_norm_len] + predict_value[0][j]
        cur_data = [cur_len, run_norm_len, boost_norm_len, predict_value[0][j]]

        cur_seq.append(cur_data)

predict_value = model.predict([cur_seq])
predict_list += predict_value[0].tolist()


real = (np.array(event_data["#2500"].values.tolist()) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
predict = (np.array(predict_list) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
#
plt.grid(True)
plt.plot(real, label="real")
plt.plot(list(range(len(real), run_len)), predict[:run_len - len(real)], linestyle="--", label="predict")
plt.legend(loc='lower right', frameon=False)
plt.show()

print(predict[run_len - len(real)] - 1)