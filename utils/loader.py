import numpy as np


def load_txt(file_path):
    """Loads .txt data into a numpy array.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        A numpy array with the loaded data.

    """

    # Loads the .txt with numpy
    array = np.loadtxt(file_path)

    return array


def load_npy(file_path):
    """Loads .npy data into a numpy array.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        A numpy array with the loaded data.

    """

    # Loads the .npy with numpy
    array = np.load(file_path)

    return array


def save_npy(array, file_path):
    """Saves numpy arrays into a .npy file.

    Args:
        array (np.array): Numpy array to be saved
        file_path (str): Path to the file to be saved.

    """

    # Saves the numpy array into a .npy file
    np.save(file_path, array)
