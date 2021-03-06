import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): actual labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    #get the four results into an array using a helper function
    results = getMetrics(actual, predictions);

    true_positives = results[0]
    true_negatives = results[1]
    false_positives = results[2]
    false_negatives = results[3]

    


    confusion_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
    return confusion_matrix
     


def getMetrics (actual, predictions):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    #if predicted to be positive
    for i in range(predictions.shape[0]):
        if predictions[i] == True:
            #predicted to be true
            if actual[i] == True:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if actual[i] == True:
                false_negatives += 1
            else:
                true_negatives += 1

    return [true_positives, true_negatives, false_positives, false_negatives]
    

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    results = getMetrics(actual, predictions)
    if(results[0] + results[1] + results[2] + results[3]) == 0:
        return 0; 
    return (results[0] + results[1])/(results[0] + results[1] + results[2] + results[3])

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """


    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    results = getMetrics(actual, predictions)

    if (results[0] + results[2]) == 0:
        precision = 0
    else:
        precision = results[0]/(results[0] + results[2])  #TP / TP + FP

    if(results[0] + results[3]) == 0:
        recall = 0
    else:
        recall = results[0] /(results[0] + results[3])  #TP / TP + TF

    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)

    if(recall + precision) == 0:
        return 0
    return (2 * recall * precision)/(recall + precision)

