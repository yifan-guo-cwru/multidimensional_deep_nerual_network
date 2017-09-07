from __future__ import print_function

import os
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # in case of warning to add this line
"""
TF_CPP_MIN_LOG_LEVEL is a TensorFlow environment variable responsible for the logs, to silence INFO logs set it to 1, 
to filter out WARNING 2 and to additionally silence ERROR logs (not recommended) set it to 3
"""

#load generated data
from data_generation import *

# keras feature
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical

def do_main_architecture():

    # Initial assignment & Tuning parameters
    Parameters_Setting = {}
    SAVE_RESULT = True
    MAX_LEN = maxlen
    BATCH_SIZE = 12
    OPTIMIZER = 'RMSprop'
    LEARNING_RATE = 0.001
    RHO = 0.9
    EPSILON = 1e-01
    LOSS_FUNCTION = 'binary_crossentropy'
    ACTIVATION_FUNCTION = 'sigmoid'
    EPOCHS = 5
    Parameters_Setting["MAX_LEN"], Parameters_Setting["BATCH_SIZE"], Parameters_Setting["OPTIMIZER"], \
    Parameters_Setting["LEARNING_RATE"], Parameters_Setting["RHO"], Parameters_Setting["EPSILON"], \
    Parameters_Setting["LOSS_FUNCTION"], Parameters_Setting["ACTIVATION_FUNCTION"], Parameters_Setting["EPOCHS"] \
        = MAX_LEN, BATCH_SIZE, OPTIMIZER, LEARNING_RATE, RHO, EPSILON, LOSS_FUNCTION, ACTIVATION_FUNCTION, EPOCHS

    # Build the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, lenblocks)))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(LSTM(128))
    #model.add(Dropout(0.2))
    # model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(lenblocks))
    model.add(Activation(ACTIVATION_FUNCTION))

    # Define the optimizer
    optimizer = RMSprop(lr=LEARNING_RATE, rho=RHO, epsilon=EPSILON)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
    model.compile(loss=LOSS_FUNCTION, optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    if SAVE_RESULT == True:

        train_predict_proba = model.predict_proba(X_train, batch_size=BATCH_SIZE)
        predict_result = model.predict(X_test, batch_size=BATCH_SIZE)
        predict_proba = model.predict_proba(X_test, batch_size=BATCH_SIZE)

        #print ("predict_classes : " + str(predict_result))
        #print ("predict_proba : " + str(predict_proba))

        predict_result[predict_result >= 0.5] = 1
        predict_result[predict_result < 0.5] = 0

        # print raw_data_matrix
        df = pandas.DataFrame(y_train)
        df.to_csv('../DATA/ANALYSIS_DATA/77district_train_real_result_2013_to_2015.csv')

        # print raw_data_matrix
        df = pandas.DataFrame(train_predict_proba)
        df.to_csv('../DATA/ANALYSIS_DATA/77district_train_proba_2013_to_2015.csv')

        # print raw_data_matrix
        df = pandas.DataFrame(y_test)
        df.to_csv('../DATA/ANALYSIS_DATA/77district_test_real_result_2013_to_2015.csv')

        # print raw_data_matrix
        df = pandas.DataFrame(predict_result)
        df.to_csv('../DATA/ANALYSIS_DATA/77district_test_predict_result_2013_to_2015.csv')

        # print raw_data_matrix
        df = pandas.DataFrame(predict_proba)
        df.to_csv('../DATA/ANALYSIS_DATA/77district_test_proba_2013_to_2015.csv')

    #Evaluate the output
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

    #print out the performance
    print ("\nSystem performance:")
    print ("Test loss : " + str(loss))
    print ("Test accuracy : " + str(accuracy))

    prediction = model.predict(X_test, batch_size=BATCH_SIZE)
    mse = np.mean(np.square(prediction - y_test), axis=1)
    score = np.mean(mse)
    print("Test MSE : " + str(score))

    #print(model.summary())
    # Save the file
    with open('Model_Summary.txt', 'a') as fh:
        # Instead of "w" use "a"(append) mode with open function
        # Pass the file handle in as a lambda function to make it callable
        fh.write('Time: ' + str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        fh.write("\nSystem Model:\n")
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        for key in Parameters_Setting:
            fh.write("%s: %f\n"%(key, Parameters_Setting[key]))
    fh.close()

    return Parameters_Setting
# do_main_architecture()