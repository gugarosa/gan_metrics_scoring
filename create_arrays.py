import argparse

import utils as u


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

    parser.add_argument('files', help='List of files to be loaded', nargs='+')

    parser.add_argument(
        '-n_samples', help='Maximum number of samples to load', type=int, default=960)

    return parser.parse_args()


if __name__ == "__main__":
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    path = args.path
    files = args.files
    n_samples = args.n_samples

    # Loading the pre-defined input files
    data = [u.load_txt(f'{path}/{f}') for f in files]

    # Parsing the data
    parsed_data = [u.parse_data(d, n_rows=n_samples) for d in data]

    # Concatenating the data into a feature vector
    concat_data = u.concat_data(parsed_data)

    # Saving data back as a numpy array
    u.save_data(concat_data, output_path=f'{path}/features.npy')
