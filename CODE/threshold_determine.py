# set up the initial threshold from training data
import numpy as np
import math
import random
import sys
import pandas

def do_threshold_determine():
    # get the file from training data
    train_proba_file = '../DATA/ANALYSIS_DATA/77district_train_proba_2013_to_2015.csv'
    train_real_label_file = '../DATA/ANALYSIS_DATA/77district_train_real_result_2013_to_2015.csv'

    # get the file of testing data
    test_proba_file = "../DATA/ANALYSIS_DATA/77district_test_proba_2013_to_2015.csv"
    test_pred_label_file = "../DATA/ANALYSIS_DATA/77district_test_predict_result_2013_to_2015.csv"
    test_real_label_file = "../DATA/ANALYSIS_DATA/77district_test_real_result_2013_to_2015.csv"

    # load data
    train_proba = pandas.read_csv(train_proba_file, engine='python').values.astype('float32')
    train_real_label = pandas.read_csv(train_real_label_file, engine='python').values.astype('float32')
    test_proba = pandas.read_csv(test_proba_file, engine='python').values.astype('float32')
    test_pred_label = pandas.read_csv(test_pred_label_file, engine='python').values.astype('float32')
    test_real_label = pandas.read_csv(test_real_label_file, engine='python').values.astype('float32')

    # refine the data
    train_proba = train_proba[0:int(train_proba.shape[0]), 1:train_proba.shape[1]]
    train_real_label = train_real_label[0:int(train_real_label.shape[0]), 1:train_real_label.shape[1]]
    test_proba = test_proba[0:int(test_proba.shape[0]), 1:test_proba.shape[1]]
    test_pred_label = test_pred_label[0:int(test_pred_label.shape[0]), 1:test_pred_label.shape[1]]
    test_real_label = test_real_label[0:int(test_real_label.shape[0]), 1:test_real_label.shape[1]]

    # data shape
    print_data_shape = False
    if print_data_shape == True:
        print("train_proba_shape: " + str(train_proba.shape))
        print("train_real_label: " + str(train_real_label.shape))
        print("test_proba: " + str(test_proba.shape))
        print("test_pred_label: " + str(test_pred_label.shape))
        print("test_real_label: " + str(test_real_label.shape))

    # Initial assignment
    common_block_size = train_proba.shape[1] # the number of community areas
    train_sample_size = train_proba.shape[0] # the number of training samples
    test_sample_size = test_proba.shape[0] # the number of testing samples
    threshold = np.zeros((1, common_block_size))

    # generate the current threshold array
    def CTA(focus_proba_array, current_threshold):
        #focus_proba_array[focus_proba_array > current_threshold] = 1
        #focus_proba_array[focus_proba_array <= current_threshold] = 0
        #current_threshold_array = focus_proba_array
        current_threshold_array = np.zeros(focus_proba_array.shape)
        for index in xrange(len(focus_proba_array)):
            if focus_proba_array[index] > current_threshold:
                current_threshold_array[index] = 1
            else:
                current_threshold_array[index] = 0
        return current_threshold_array

    # calculate the mean square error
    def MSE(focus_array, current_threshold_array):
        #mse = np.mean(np.square(focus_array - current_threshold_array), axis=1)
        #mse = np.array((focus_array + current_threshold_array)%2)
        guess = (focus_array != current_threshold_array)
        score = guess.sum()
        return score

    # the test case function
    def test_case1(current_threshold_array = np.array([[0],[0],[1]])):
        focus_array = np.array([[1],[0],[1]])
        #print(focus_array)
        #print(focus_array - current_threshold_array)
        #print(np.mean(np.square(focus_array - current_threshold_array), axis=1))
        score = MSE(focus_array, current_threshold_array)
        print(score)

    def test_case2():
        focus_proba_array = np.array([[0.45], [0.55], [0.49]])
        current_threshold = 0.5
        current_threshold_array = CTA(focus_proba_array, current_threshold)
        return current_threshold_array

    # print(test_case1(test_case2()))

    total_score = 0
    # while loop to find the best threshold for each column
    for block_number in xrange(common_block_size):
        focus_proba_array = test_proba[:,block_number].astype('float32')
        #print(focus_proba_array)
        focus_array = test_real_label[:,block_number]
        #print(focus_array)
        step = 1e-4
        initial_point = np.mean(focus_proba_array)
        final_point = np.mean(focus_proba_array) + np.std(focus_proba_array) #np.max(focus_proba_array)
        print(np.mean(focus_proba_array))
        print(np.std(focus_proba_array))
        print("Final Point: " + str(final_point))
        #print(final_point)
        #print("# of 1: " + str(np.sum(focus_array)))
        #test_threshold = np.mean(focus_proba_array)
        #current_threshold_array = CTA(focus_proba_array, initial_point)
        #print(current_threshold_array)
        #score = MSE(focus_array, current_threshold_array)
        #print("final point current score: " + str(score))
        #test_threshold = 0.4
        #current_threshold_array = CTA(focus_proba_array, test_threshold)
        #score = MSE(focus_array, current_threshold_array)
        #print("random current score: " + str(score))
        point = initial_point
        threshold_point = point
        threshold_point_score = test_sample_size
        current_score = 10000000
        while (point < final_point):
            #print("point: " + str(point))
            current_threshold_array = CTA(focus_proba_array, point)
            #print("# of 1 in current_threshold_array: " + str(np.sum(current_threshold_array)))
            #test = np.abs(focus_array - current_threshold_array)
            current_score = MSE(focus_array, current_threshold_array)
            #print("current score: " + str(current_score))
            if current_score <= threshold_point_score:
                threshold_point = point
                threshold_point_score = current_score
            point = point + step
        print("threshold point for block_number " + str(block_number+1) + " is: " + str(threshold_point))
        print("score for block_number " + str(block_number+1) + " is: " + str(threshold_point_score))
        total_score = total_score + threshold_point_score
        threshold[0][block_number] = threshold_point

    # collect the data of threshold
    df = pandas.DataFrame(threshold)
    df.to_csv('../DATA/ANALYSIS_DATA/77district_train_threshold_2013_to_2015.csv')

    # generate new prediction data
    test_threshold_pred_label = np.zeros(test_pred_label.shape)
    for block_number in xrange(common_block_size):
        current_threshold_array = CTA(test_proba[:,block_number], threshold[0][block_number])
        #test_threshold_pred_label[:,block_number] = current_threshold_array
        for index in xrange(len(current_threshold_array)):
            test_threshold_pred_label[index][block_number] = current_threshold_array[index]

    # collect the data of threshold
    df = pandas.DataFrame(test_threshold_pred_label)
    df.to_csv('../DATA/ANALYSIS_DATA/77district_test_refine_predict_result_2013_to_2015.csv')

    print("Verify wrong prediction number: " + str(total_score))
    print("Verify accuracy: " + str(1-(1.0*total_score)/(test_sample_size*block_number)))

#do_threshold_determine()