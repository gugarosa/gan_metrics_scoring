import argparse

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Loads features, targets .npy files and fits a SVM.')

    parser.add_argument(
        'features', help='Path to the features .npy file', type=str)

    parser.add_argument(
        'targets', help='Path to the targets .npy file', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    feature_array = args.features
    target_array = args.targets

    # Loads the arrays
    features = l.load_npy(feature_array)
    targets = l.load_npy(target_array)

    # Dividing data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=42)

    # Instanciating classifier
    clf = SVC(C=100, kernel='rbf', probability=True)

    # Fitting classifier
    clf.fit(x_train, y_train)

    # Scoring data with classifier
    print(f'Accuracy: {clf.score(x_test, y_test)}')

    #
    sample = [[0.6, 0.1, 0.9]]

    # Predicting new data
    pred = clf.predict(sample)
    prob = clf.predict_proba(sample)

    print(f'Sample: {sample} | Class: {pred} | Probs: {prob}')