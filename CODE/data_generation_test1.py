# basic mathematical library
import numpy as np
import math
import pandas

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))
print("The shape of selected x_train: " + str(x_train.shape))
print("The shape of selected y_train: " + str(y_train.shape))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))
print("The shape of selected x_val: " + str(x_train.shape))
print("The shape of selected y_val: " + str(y_train.shape))
