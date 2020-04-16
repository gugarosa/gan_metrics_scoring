import argparse

import utils.data as d
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Gets an input data and creates Numpy-based arrays.')

    parser.add_argument(
        'path', help='Path to the folder containing the data', type=str)

    parser.add_argument(
        'files', help='List of files to be loaded', nargs='+')

    parser.add_argument(
        '-n_samples', help='Maximum number of samples to load', type=int, default=960)

    parser.add_argument(
        '-normalize', help='Whether data should be normalized or not', type=bool, default=True)

    parser.add_argument(
        '-outlier', help='Whether outliers should be capped or not', type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    path = args.path
    files = args.files
    n_samples = args.n_samples
    normalize = args.normalize
    outlier = args.outlier

    # Loading the pre-defined input files
    arrays = [l.load_txt(f'{path}/{f}') for f in files]

    # Extracting the feature column from arrays with desired number of samples
    feature_arrays = [d.extract_feature(a, n_samples) for a in arrays]

    # Creating a feature array from the list of feature arrays
    feature_array = d.feature_vector(feature_arrays, normalize, outlier)

    print(feature_array.shape)

    # print(m.remove_outliers(concat_data[:, 1], 1.5).max())

    # Saving data back as a numpy array
    # s.save_data(concat_data, output_path=f'{path}/features.npy')
