import json
import os
from model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


idol_value: pd.DataFrame = pd.read_csv('./anniversary/reform_idol_value.csv')

with open("./data_mean_std.json", "r", encoding='utf-8') as file:
    mean_std_dict = json.load(file)

with open("./run_time_len.json", "r", encoding='utf-8') as file:
    time_dict = json.load(file)

with open("./cards/event.json", "r", encoding='utf-8') as file:
    event_idol = json.load(file)

data_path = './tour/reform_csv/'
data_csv_list = os.listdir(data_path)
data_csv_list = [int(item[:-4]) for item in data_csv_list]
data_csv_list.sort()
data_arr = []

for cur_target in data_csv_list:
    rank_idol = event_idol[str(cur_target)]['ranking'] - 1
    rank_vector = idol_value.loc[rank_idol].values.tolist()

    point_idol = event_idol[str(cur_target)]['point'] - 1
    point_vector = idol_value.loc[point_idol].values.tolist()

    event_data: pd.DataFrame = pd.read_csv(data_path + str(cur_target) + '.csv')

    run_len = time_dict[str(cur_target)]["run_time_len"]
    boost_len = time_dict[str(cur_target)]["boost_time_len"]

    run_norm_len = (float(run_len) - time_dict["run_time_mean"]) / time_dict["run_time_std"]
    boost_norm_len = (float(boost_len) - time_dict["boost_time_mean"]) / time_dict["boost_time_std"]

    point_data = []

    for i in range(len(event_data)):
        try:
            event_value = event_data.loc[i]["#2500"]
            #event_value = event_data.loc[i].values.tolist()
        except Exception as e:
            print(i, run_len, len(event_data))
            break
        cur_len = float(i + 1) / run_len
        # cur_data = [cur_len, run_norm_len, boost_norm_len] + event_value # + rank_vector # + point_vector
        cur_data = [cur_len, run_norm_len, boost_norm_len, event_value] #+ rank_vector  #+ point_vector

        point_data.append(cur_data)

    data_arr.append(point_data)

train_x = []
train_y = []
test_x = []
test_y = []

seq_length = 144

for idx in range(0, len(data_arr) - 1):
    x, y = multivariate_data(data_arr[idx], 3, seq_length)
    train_x.extend(list(x))
    train_y.extend(list(y))

test_x, test_y = multivariate_data(data_arr[0], 3, seq_length)
print(len(cur_data))
model = MLTD_model(data_dim=len(cur_data), seq_length=seq_length)
model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.00225))
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=1000, epochs=100, shuffle=True)

plt.plot(history.history['loss'], 'y', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.show()

model.save('2500_model.h5')

model.evaluate(test_x, test_y)
result_y = model.predict(test_x)


real = (np.array(test_y[0]) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
predict = (np.array(result_y[0]) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
plt.grid(True)
plt.plot(real, label="real")
plt.plot(predict, linestyle="--", label="predict")
plt.legend(loc='lower right', frameon=False)
plt.show()


# test_x, test_y = multivariate_data(data_arr[-2], 4, seq_length)
# result_y = model.predict(test_x)
# real = (np.array(test_y) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
# predict = (np.array(result_y) * mean_std_dict["#2500"]["data_std"]) + mean_std_dict["#2500"]["data_mean"]
# plt.grid(True)
# plt.plot(real, label="real")
# plt.plot(predict, linestyle="--", label="predict")
# plt.legend(loc='lower right', frameon=False)
# plt.show()