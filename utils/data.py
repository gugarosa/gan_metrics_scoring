import numpy as np

import utils.math as m


def extract_feature(array, n_samples=1):
    """Extracts the feature column from an array.

    Note that this method expects the first column to the the row identifier
    and the second column to be the feature.

    Args:
        n_samples (int): Maximum number of samples.

    Returns:
        The extracted feature column out of an array

    """

    # Parses the data with n_samples and using the last column as the feature column
    feature = array[:n_samples, -1]

    return feature


def feature_vector(arrays, use_normalization=True, use_capped_outliers=True):
    """Creates a feature vector from a list of numpy arrays.

    Args:
        arrays (list): List of numpy arrays.
        use_normalization (bool): Whether final array should be normalized or not.
        use_capped_outliers (bool): Whether final array should have capped outliers or not.

    Returns:
        A numpy array with (n_samples, n_features) shape.

    """

    print('Creating feature vector ...')

    # Concatenates the arrays into a single array
    vector = np.stack(arrays)

    # Transposes its dimensions to (n_samples, n_features)
    vector = np.transpose(vector, [1, 0])

    # Check the normalization boolean
    if use_normalization:
        # Normalizes the array
        vector = vector / vector.max(axis=0)

    # Checks the capped outliers boolean
    if use_capped_outliers:
        # Caps the array outliers
        vector = m.cap_outliers(vector)

    print(
        f'Shape: {vector.shape} | Normalized: {use_normalization} | Capped Outliers: {use_capped_outliers}')

    return vector
