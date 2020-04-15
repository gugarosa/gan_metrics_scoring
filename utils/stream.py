import numpy as np


def load_txt(file_path):
    """Loads raw .txt data into a numpy array.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        A numpy array with the loaded data.

    """

    # Loads the data using numpy
    data = np.loadtxt(file_path)

    return data


def parse_data(data, n_rows):
    """Parses the data into desired amount of samples.

    Note that this method expects the first column to the the row identifier
    and the second column to be the feature.

    Args:
        n_rows (int): Maximum number of rows (samples)

    Returns:
        A parsed numpy array.

    """

    # Parses the data with n_rows and using the last column as the feature
    parsed_data = data[:n_rows, -1]

    return parsed_data


def concat_data(parsed_data):
    """Concatenates a list of parsed data into a single numpy array.

    Args:
        parsed_data (list): List of already parsed numpy arrays.

    Returns:
        A single numpy array.
    """

    # Concatenates the data into a single array
    concat_data = np.stack(parsed_data)

    # Transposes its dimensions to (n_samples, n_features)
    concat_data = np.transpose(concat_data, [1, 0])

    return concat_data


def save_data(concat_data, output_path=''):
    """Saves concatenated data into a numpy array.

    Args:
        concat_data (np.array): Concatenated numpy array.
        output_path (str): Path to save the numpy array.

    """

    # Saves the data to a numpy array
    np.save(output_path, concat_data)


def load_data(input_path='', normalize=False):
    """Loads already-saved numpy array.

    Args:
        input_path (str): Path to load the numpy array.
        normalize (bool): Whether data should be normalized or not.

    """

    # Loads the already-saved numpy array
    array = np.load(input_path)

    # Check if it is supposed to be normnalized
    if normalize:
        # Normalizes the data
        array = array / array.max(axis=0)

    return array
