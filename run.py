import utils as u

# Defines an static path folder to the data
STATIC_PATH = 'data/1/'

# Defines the files to be loaded
FILES = [STATIC_PATH + 'fid.txt', STATIC_PATH + 'mse.txt', STATIC_PATH + 'ssim.txt']

# Maximum number of samples for parsing the data
MAX_SAMPLES = 10


if __name__ == "__main__":
    # Loading the pre-defined input files
    data = [u.load_txt(f) for f in FILES]

    # Parsing the data
    parsed_data = [u.parse_data(d, n_rows=MAX_SAMPLES) for d in data]

    # Concatenating the data into a feature vector
    x = u.concat_data(parsed_data)
