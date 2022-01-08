'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
PHUONG NGUYEN NGOC
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''

    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None
        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape
        self.class_priors = np.zeros(self.num_classes)
        self.class_likelihoods = np.zeros((self.num_classes, data.shape[1]))
        for i in range(self.num_classes):
            class_row = np.where(y == i)[0]
            self.class_priors[i] = len(class_row)/y.shape[0]
            likelihood_numerator = np.sum(data[np.ix_(class_row)], axis=0) + 1
            likelihood_denominator = np.sum(
                data[np.ix_(class_row)]) + data.shape[1]
            self.class_likelihoods[i] = likelihood_numerator / \
                likelihood_denominator

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Process test samples one-by-one.
        - For each, we want to compute the log of the numerator of the posterior:
        - (a) Use matrix-vector multiplication (or the dot product) with the log of the likelihoods
          and the test sample (transposed, but no logarithm taken)
        - (b) Add to it the log of the priors
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (use argmax)
        '''
        log_post = np.log(self.class_priors) + \
            data@np.log(self.class_likelihoods.T)
        predicted_data = np.argmax(log_post, axis=1)
        return predicted_data

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        correct_pred = (y == y_pred)
        true_pred_index = np.where(correct_pred == True)
        true_pred_total = len(true_pred_index[0])
        return true_pred_total/y.shape[0]

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.
        num_classes = np.unique(y).shape[0]
        confusion_matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                gt = np.where(y == i)[0]
                pred = y_pred[np.ix_(gt)]
                confusion_matrix[i, j] = len(np.where(pred == j)[0])
        return confusion_matrix

    def k_fold_cross_validation(self, features, y, k, shuffle=False):
        k_acc = np.zeros(k)
        if shuffle:
            inds = np.arange(features.shape[0])
            np.random.shuffle(inds)
            features = features[inds]
            y = y[inds]

        k_features = np.split(features, k, axis=0)
        k_y = np.split(y, k)

        for i in range(k):
            if i == k-1:
                train_x = np.vstack(k_features[:i])
                train_y = np.concatenate(k_y[:i])
            elif i == 0:
                train_x = np.vstack(k_features[1:])
                train_y = np.concatenate(k_y[1:])
            else:
                train_x = np.vstack(
                    (np.vstack(k_features[:i]), np.vstack(k_features[i+1:])))
                train_y = np.concatenate(
                    (np.concatenate(k_y[:i]), np.concatenate(k_y[i+1:])))
            test_x = k_features[i]
            test_y = k_y[i]
            self.train(train_x, train_y)
            y_pred = self.predict(test_x)
            acc = self.accuracy(test_y, y_pred)
            k_acc[i] = acc
        return np.mean(k_acc), k_acc
