import argparse

import numpy as np

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Digitizes a numpy array into intervals in order to create targets.')

    parser.add_argument(
        'input', help='Path to the .npy file', type=str)

    parser.add_argument(
        '-n_bins', help='Number of intervals to digitize', type=int, default=5)

    return parser.parse_args()

if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_array = args.input
    n_bins = args.n_bins

    # Loads the array
    features = l.load_npy(input_array)

    # Gathering minimum and maximum feature values
    min_features = features.min(axis=0)
    max_features = features.max(axis=0)
    
    # Pre-allocating targets array
    y = np.zeros((features.shape[0], features.shape[1]), dtype=np.int)

    print('Creating targets ...')

    # For every possible feature
    for i, (min_f, max_f) in enumerate(zip(min_features, max_features)):
        # Creating equally-spaced intervals
        bins = np.linspace(min_f, max_f, n_bins+1)

        # If iteration corresponds to FID or MSE metric
        if i == 0 or i == 1:
            # Digitizing the features array with flipped intervals
            y[:, i] = np.digitize(features[:, i], np.flip(bins), right=True)

        # If not
        else:
            # Digitizing the features array
            y[:, i] = np.digitize(features[:, i], bins)

    # Gathering most voted `y` along the features
    targets = np.asarray([(np.argmax(np.bincount(y[i, :]))) for i in range(features.shape[0])])

    print(f'Labels, Counts: {np.unique(targets, return_counts=True)}')

    # Saving targets array as a .npy file
    l.save_npy(targets, f'targets.npy')
