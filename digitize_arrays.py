import argparse

import numpy as np

import utils.stream as s


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Digitizes array into intervals in order to create their target.')

    parser.add_argument(
        'input', help='Path to the saved numpy array', type=str)

    parser.add_argument(
        '-n_bins', help='Number of intervals to digitize', type=int, default=5)

    parser.add_argument(
        '-normalize', help='Whether data should be normalized or not after loading', type=bool, default=False)

    return parser.parse_args()

if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_array = args.input
    n_bins = args.n_bins
    normalize = args.normalize

    # Loads the array
    features = s.load_data(input_array, normalize=normalize)

    # Gathering minimum and maximum feature values
    min_features = features.min(axis=0)
    max_features = features.max(axis=0)

    # For every possible feature
    for i, (min_f, max_f) in enumerate(zip(min_features, max_features)):
        # Creating equally-spaced intervals
        bins = np.linspace(min_f, max_f, n_bins + 1)

        print(bins)

        # If iteration corresponds to MSE's metric
        if i == 1:
            # Digitizing the features array with flipped intervals
            y = np.digitize(features[:, i], np.flip(bins))

        # If not
        else:
            # Digitizing the features array
            y = np.digitize(features[:, i], bins)

        print(y)