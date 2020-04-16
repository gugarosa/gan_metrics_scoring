import numpy as np


def cap_outliers(array, c=1.5):
    """Caps the outliers from a numpy array.

    Args:
        array (np.array): Array to be capped.
        c (float): Constant used to calculate the IQR.

    Returns:
        A numpy array with capped outliers.

    """

    # Calculating the upper quartile
    upper_quartile = np.percentile(array, 75, axis=0)

    # Calculating the lower quartile
    lower_quartile = np.percentile(array, 25, axis=0)

    # Calculating the IQR
    IQR = (upper_quartile - lower_quartile) * c

    # Calculating the capping range
    cap_range = [(l - i, u + i)
                 for l, u, i in zip(lower_quartile, upper_quartile, IQR)]

    # Iterating over all possible variables
    for i, cap in enumerate(cap_range):
        # Clipping column with corresponding lower and upper values
        array[:, i] = np.clip(array[:, i], cap[0], cap[1])

    return array
