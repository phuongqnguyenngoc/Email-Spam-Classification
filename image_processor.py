import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np
import cv2
from pca_cov import PCA_COV
import math


def preprocess_single_image(filename):
    '''
    Resize image to shape (64,64), grayscale, then flatten

    Parameters:
    -----------
    filename: string. path to the image

    Returns:
    -----------
    image: ndarray shape(4096,)
        Processed image
    '''
    image = cv2.imread(filename)
    image = cv2.resize(image, (64, 64))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.reshape(image, (image.shape[0]*image.shape[1]))
    return image


def preprocess_directory(image_dir):
    '''
    Process and label images in a directory

    Parameter:
    -----------
    image_dir: path to the base directory

    Returns:
    -----------
    features: ndarray. shape(
        num_data_points, processed_image_height*processed_image_width)
        Matrix of processed image with each row being an image vector
    y: ndarray. shape (num_data_points, )
        Labels corresponding to images in the rows of features matrix
    num_fruits: int
        number of data points
    '''
    features = []
    y = []
    num_fruits = 0
    classes = os.listdir(image_dir)

    for c in classes:
        class_path = os.path.join(image_dir, c)
        if ".DS_Store" in class_path:
            continue
        fruits = os.listdir(class_path)
        for f in fruits:
            num_fruits += 1
            fruit = os.path.join(class_path, f)
            if ".DS_Store" in fruit:
                continue
            features.append(preprocess_single_image(fruit))
            if c == "Banana":
                y.append(0)
            elif c == "Orange":
                y.append(1)
            elif c == "Peach":
                y.append(2)

    features = np.array(features)
    y = np.array(y)
    return features, y, num_fruits


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:
    no_train_samples = features.shape[0]
    cutoff = int(no_train_samples*(1-test_prop))
    x_train = features[:cutoff]
    y_train = y[:cutoff]
    inds_train = inds[:cutoff]

    x_test = features[cutoff:]
    y_test = y[cutoff:]
    inds_test = inds[cutoff:]

    return(x_train, y_train, inds_train, x_test, y_test, inds_test)


def retrieve_image(inds, image_path):
    '''Obtain the ndarray of read image at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: int
        The index of the image. Indices are counted from 0 to num_images-1
        (counting does NOT reset when switching to emails of another class).
    image_path: str. Relative path to the image dataset base folder.

    Returns:
    -----------
    img: ndarray. shape(img.height, img.width)
    '''
    classes = os.listdir(image_path)
    count = 0

    for c in classes:
        class_path = os.path.join(image_path, c)
        if ".DS_Store" in class_path:
            continue
        class_images = os.listdir(class_path)
        # count += len(emails) - 1
        if count + len(class_images) < inds:
            count += len(class_images)
        else:  # count + len(emails) >= inds
            # print(f'count {count}')
            # print(f'path {class_images[inds-count]}')
            filename = os.path.join(class_path, class_images[inds-count])
            if ".DS_Store" in filename:
                filename = os.path.join(class_path, class_images[inds-count+1])
            print(f'\n{filename}\n')
            img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            # img = np.reshape(
            #     img, (int(math.sqrt(img.shape[0])), int(math.sqrt(img.shape[0]))))
            # plt.imshow(img)
            # plt.show()
            return img


def PCA_project(pca_obj, top_k):
    '''
    project the data of the pca_obj onto the PCA space of top_k

    Parameters:
    -----------
    pca_obj: object of class PCA_COV
    top_k: int     number of PCs to keep

    Returns:
    -----------
    projected_data: ndarray. shape=(pca_obj.num_samps, top_k).
        Data that is projected onto the PCA space
    '''
    projected_data = pca_obj.pca_project(top_k)
    return projected_data


def plot_images(dataset, subtitle, n_row, n_col, title=""):
    '''
    Generate subplot of image with subtitle for each row

    Parameters:
    -----------
    dataset: ndarray. shape(num_images, images.shape[0]*images.shape[1])
        Matrix of flatten image vectors
    subtitle: Python list of strings
        List of title for each row
    n_row: int
        Number of rows of plots
    n_col: int
        Number of columns of plots
    title: string
        Title of the big plot
    '''
    # subtitle is a python list of title for each row
    fig, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=n_row, ncols=1)
    fig.suptitle(title, ha="center", va="baseline", fontsize=50)
    for row, big_ax in enumerate(big_axes, start=0):
        big_ax.set_title(subtitle[row], fontsize=30)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0),
                           top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1, n_row*n_col+1):
        ax = fig.add_subplot(n_row, n_col, i)
        img = np.reshape(dataset[i-1], (int(
            math.sqrt(dataset[0].shape[0])), int(math.sqrt(dataset[0].shape[0]))))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, cmap=plt.get_cmap('gray'))
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.show()
