import pandas as pd
from values import *
from model import *
import matplotlib.pyplot as plt


def multivariate_data(dataset, target, window_size):
    data = []
    labels = []
    dataset = np.array(dataset)
    for i in range(window_size, len(dataset)):
        indices = range(i - window_size, i)
        data.append(dataset[indices].tolist())

        labels.append(dataset[i][target])

    return data, labels


len_mean = 361.0952380952381
len_std = 30.97080038020452

data_mean = 57661.34498219702
data_std = 57279.672070748755

data_arr = []

for idx in theater:
    point_data = []

    with open('./data/theater/' + str(idx) + '.json') as raw_file:
        json_data = json.load(raw_file)
        json_data = json_data[1]["data"]

        data_len = len(json_data)
        norm_len = (float(data_len) - len_mean) / len_std

        for j, item in enumerate(json_data):
            point = (item["score"] - data_mean) / data_std
            cur_len = float(j + 1) / data_len

            point_data.append([point, cur_len, norm_len])

    data_arr.append(point_data)
    point_data = pd.DataFrame(point_data)
    point_data.to_csv('./data/theater/' + str(idx) + '.csv', header=False, index=False)

train_x = []
train_y = []
test_x = []
test_y = []
for idx in range(0, len(data_arr) - 1):
    x, y = multivariate_data(data_arr[idx], 0, 48)
    train_x.extend(list(x))
    train_y.extend(list(y))

test_x, test_y = multivariate_data(data_arr[len(data_arr) - 1], 0, 48)


model = MLTD_model()
model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.0005))
history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=500, epochs=100, shuffle=True)

plt.plot(history.history['loss'], 'y', label='train loss')
plt.plot(history.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.show()

model.save('2500_model.h5')

model.evaluate(test_x, test_y)

result_y = model.predict(test_x)
plt.grid(True)
plt.plot((np.array(test_y) * data_std) + data_mean, label="real")
plt.plot((np.array(result_y) * data_std) + data_mean, linestyle="--", label="predict")
plt.legend(loc='lower right', frameon=False)
plt.show()


'''
test_val = test_x[0].copy()

result = []
for i in range(1, len(test_x)):
    print(i)
    y = model.predict([test_val])
    result.append(y[0][0])
    new_val = test_x[i][47].copy()
    new_val[0] = y[0][0].item()
    test_val.append(new_val)
    test_val.pop(0)
'''

'''
    theater_list.append(point_data)

theater_list = np.array(theater_list)

df = pd.array(point_arr)
mean = df.mean()
std = df.std()

pre_theater_list = []
for arr in theater_list:
    pre_theater_list.append((arr - mean) / std)


plt.grid(True)
plt.yscale('log')
for arr in theater_list:
    plt.plot(arr)

plt.title('theater event in2500')

plt.xlabel('x')
plt.ylabel('pt')
plt.xlim(0, 400)

plt.show()
'''