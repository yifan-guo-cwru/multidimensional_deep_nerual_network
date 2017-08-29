# basic mathematical library
import numpy as np
import math
import pandas

file_name = '../DATA/RAW_DATA/multihot_vector_77district_2013_to_2015.csv'
# dataframe = pandas.read_csv('../crime_data/raw_data_matrix_9district_multihot_2013_to_2015.csv', usecols = [1,2,3,4,5,6,7,8,9], engine='python')
dataframe = pandas.read_csv(file_name, engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
select_ratio = 1 / 3.0
dataset = dataset[0:int(math.ceil(select_ratio * dataset.shape[0])), 1:dataset.shape[1]]
# print (dataset[0,:])
print("dataset shape: " + str(dataset.shape))

maxlen = 1
step = 1
sentences = []
next_block = []

lenblocks = len(dataset[0, :])
lentimeslots = len(dataset[:, 0])
print('lenblocks: ', lenblocks)
print('lentimeslots: ', lentimeslots)

for i in range(0, lentimeslots - maxlen, step):
    sentences.append(dataset[i:i + maxlen, :])
    next_block.append(dataset[i + maxlen, :])

print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, lenblocks))
y = np.zeros((len(sentences), lenblocks))

for i in range(0, lentimeslots - maxlen):
    for j in range(0, maxlen):
        for q in range(0, lenblocks):
            X[i, j, q] = dataset[i + j, q]
    for p in range(0, lenblocks):
        y[i, p] = dataset[i + maxlen, p]

print('Build model...')

# Define the test data and training data
ratio = 0.7  # the ratio of data assigned as training data
X_train = X[0:int(math.ceil(ratio * X.shape[0])), :, :]
y_train = y[0:int(math.ceil(ratio * y.shape[0])), :]
# y_train_binary = to_categorical(y_train)
print('X_train.shape:',X_train.shape)
print('y_train.shape:',y_train.shape)
X_test = X[int(math.ceil((ratio) * X.shape[0])):int(X.shape[0]), :, :]
y_test = y[int(math.ceil((ratio) * y.shape[0])):int(y.shape[0]), :]
##y_test_binary = to_categorical(y_test)