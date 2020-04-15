import argparse

import matplotlib.pyplot as plt

import utils.stream as s

def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Loads a pre-saved numpy array and creates its boxplot.')

    parser.add_argument(
        'input', help='Path to the saved numpy array', type=str)

    parser.add_argument(
        '-normalize', help='Whether data should be normalized or not after loadinf', type=bool, default=False)

    return parser.parse_args()

if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    input_array = args.input
    normalize = args.normalize

    # Loads the array
    features = s.load_data(input_array, normalize=normalize)

    # Gathers the number of features
    n_features = features.shape[1]

    # Creating a matplotlib figure
    fig = plt.figure()

    # For every possible column
    for i in range(n_features):
        # Defines the subplot
        plt.subplot(1, n_features, i+1)

        # Creating the boxplot
        plt.boxplot(features[:, i], notch=True)

        # Setting up the ticks
        plt.xticks([1], [f'x[{i}]'])

    # Displaying the plot
    plt.show()