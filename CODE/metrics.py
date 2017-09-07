"""
To calculate the metrics to test system performance
"""

import numpy as np
import pandas

def load_data():
    # get the file of testing data
    test_threshold_pred_label_file = '../DATA/ANALYSIS_DATA/77district_test_refine_predict_result_2013_to_2015.csv'
    test_real_label_file = "../DATA/ANALYSIS_DATA/77district_test_real_result_2013_to_2015.csv"

    # load data
    test_threshold_pred_label = pandas.read_csv(test_threshold_pred_label_file, engine='python').values.astype('float32')
    test_real_label = pandas.read_csv(test_real_label_file, engine='python').values.astype('float32')

    # refine the data
    test_threshold_pred_label = np.array(test_threshold_pred_label[0:int(test_threshold_pred_label.shape[0]),
                                         1:test_threshold_pred_label.shape[1]])
    test_real_label = np.array(test_real_label[0:int(test_real_label.shape[0]), 1:test_real_label.shape[1]])

    return test_threshold_pred_label, test_real_label

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + np.finfo(np.float32).eps)
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.finfo(np.float32).eps)
    return precision

def f1(precision_score, recall_score):
    #precision_score = precision(y_true, y_pred)
    #recall_score = recall(y_true, y_pred)
    return 2*((precision_score*recall_score)/(precision_score+recall_score))

def metrics():
    test_threshold_pred_label, test_real_label = load_data()
    y_true = test_real_label
    y_pred = test_threshold_pred_label
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    f1_score = f1(precision_score, recall_score)
    true_positives, false_positives, false_negatives, true_negatives = tp_fp_fn_tn(y_true, y_pred)
    return true_positives, false_positives, false_negatives, true_negatives, precision_score, recall_score, f1_score

def tp_fp_fn_tn(y_true, y_pred):
    (row_number, col_number) = np.shape(y_true)
    total_number = row_number*col_number
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    false_positives = predicted_positives - true_positives
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    false_negatives = possible_positives - true_positives
    true_negatives = total_number-true_positives-false_positives-false_negatives
    return true_positives, false_positives, false_negatives, true_negatives

def do_metrics():
    true_positives, false_positives, false_negatives, true_negatives, precision_score, recall_score, f1_score = metrics()
    print("true_positives, false_positives, false_negatives, true_negatives: %s, %s, %s, %s"
          % (true_positives, false_positives, false_negatives, true_negatives))
    print("true_positive_rate: " + str(true_positives/(true_positives+false_negatives)))
    print("true_negative_rate: " + str(true_negatives / (true_negatives + false_positives)))
    print("Overall system accuracy: "
          + str((true_positives+true_negatives)/(true_positives+false_positives+false_negatives+true_negatives)))
    print("precision_score: " + str(precision_score))
    print("recall_score: " + str(recall_score))
    print("f1_score: " + str(f1_score))

#do_metrics()