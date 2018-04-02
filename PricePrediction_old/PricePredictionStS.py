from __future__ import division, print_function, absolute_import

from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import tflearn

from PricePrediction_old.DataDownload import get_close_prices

# create train and validation data

input_seq_len = 32
output_seq_len = 6
training_points_num = 10000
validation_points_num = 600
validation_fraction = 0.08

time_cprice_data = get_close_prices("USDT_ETH", "1800")
transposed_data = time_cprice_data.T
cprice_data = transposed_data[1]
print(len(cprice_data))
sleep(3)
time_data = transposed_data[0]

random_low_train = input_seq_len + 1
random_high_train = int(len(cprice_data) - (len(cprice_data) * validation_fraction))
x_train = []
y_train = []
for _ in range(training_points_num):
    time = np.random.randint(random_low_train, random_high_train)
    low_time = time-input_seq_len
    x_train.append(cprice_data[low_time: time])
    y_high_time = time + output_seq_len
    y_train.append(cprice_data[time: y_high_time])

random_low_test = int(len(cprice_data) - (len(cprice_data) * validation_fraction))
random_high_test = len(cprice_data) - output_seq_len
x_test = []
y_test = []
for _ in range(validation_points_num):
    time = np.random.randint(random_low_test, random_high_test)
    low_time = time - input_seq_len
    x_test.append(cprice_data[low_time: time])
    y_high_time = time + output_seq_len
    y_test.append(cprice_data[time: y_high_time])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

# max_value = np.max([np.max(x_train), np.max(y_train), np.max(x_test), np.max(y_test)])
max_value = np.max(cprice_data)

x_train = x_train / max_value
y_train = y_train / max_value
x_test = x_test / max_value
y_test = y_test / max_value

x_train = np.reshape(x_train, (-1, 1, input_seq_len))
y_train = np.reshape(y_train, (-1, output_seq_len))
x_test = np.reshape(x_test, (-1, 1, input_seq_len))
y_test = np.reshape(y_test, (-1, output_seq_len))

# build network

net = tflearn.input_data(shape=[None, 1, input_seq_len])
net = tflearn.lstm(net, 512, return_seq=True)
net = tflearn.lstm(net, 512, return_seq=False)
net = tflearn.dropout(net, 0.7)
net = tflearn.fully_connected(net, output_seq_len, activation='leaky_relu')
net = tflearn.regression(net, optimizer='adam', loss='mean_square', learning_rate=0.001, name="output1")

# run training

model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(x_train, y_train, n_epoch=100, validation_set=(x_test, y_test), batch_size=50, show_metric=True)

# backtest

# create backtest dataset
x_backtest = []
y_backtest = []
for time in range(random_low_test, random_high_test):
    low_time = time - input_seq_len
    x_backtest.append(cprice_data[low_time: time])
    y_high_time = time + output_seq_len
    y_backtest.append(cprice_data[time: y_high_time])
x_backtest = np.asarray(x_backtest)
y_backtest = np.asarray(y_backtest)
x_backtest = x_backtest / max_value
y_backtest = y_backtest / max_value
x_backtest = np.reshape(x_backtest, (-1, 1, input_seq_len))
y_backtest = np.reshape(y_backtest, (-1, output_seq_len))

# run backtest

pred_price = model.predict(x_backtest)

# plot the predicted against true data

same_movement_vector = []

for i in range(1, output_seq_len):
    same_movement = 0
    for bt, pp in zip(y_backtest, pred_price):
        if (bt[0] - bt[i] > 0 and pp[0] - pp[i] > 0) or (bt[0] - bt[i] < 0 and pp[0] - pp[i] < 0):
            same_movement += 1
    same_movement_vector.append(same_movement)

y_bt_upscaled = y_backtest[:, -1] * max_value
y_pp_upscaled = pred_price[:, -1] * max_value
timestamp_data = time_data[-len(y_pp_upscaled):]
reshaped_y_pp_upscaled = np.ravel(y_pp_upscaled)
diff = reshaped_y_pp_upscaled - y_bt_upscaled

num = 0
for i in same_movement_vector:
    num += 1
    print("Same direction movement_" +str(num)+ ": ", i / len(reshaped_y_pp_upscaled))

print("Max of differences: ", np.max(np.abs(diff)))
print("Mean of differences: ", np.mean(diff))
print("Median of differences: ", np.median(diff))

plt.subplot(121)
plt.gca().set_title("Real prices(blue) - Predicted prices(orange)")
plt.plot(timestamp_data, y_bt_upscaled)
plt.plot(timestamp_data, reshaped_y_pp_upscaled)
plt.subplot(122)
plt.gca().set_title("Differences between the real and the predicted prices")
plt.plot(timestamp_data, diff)
plt.show()
