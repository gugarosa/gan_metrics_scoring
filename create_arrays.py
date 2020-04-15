import argparse

import utils.math as m
import utils.stream as s


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
        '-outlier', help='Whether outliers should be capped or not', type=bool, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    path = args.path
    files = args.files
    n_samples = args.n_samples
    outlier = args.outlier

    # Loading the pre-defined input files
    data = [s.load_txt(f'{path}/{f}') for f in files]

    # Parsing the data
    parsed_data = [s.parse_data(d, n_rows=n_samples) for d in data]

    # Concatenating the data into a feature vector
    concat_data = s.concat_data(parsed_data)

    print(m.remove_outliers(concat_data[:, 1], 1.5).max())

    # Saving data back as a numpy array
    s.save_data(concat_data, output_path=f'{path}/features.npy')
