# Evaluating Artificial Images Through Score-based Classifications

*This repository holds all the necessary code to run the very-same experiments described in the paper "Evaluating Artificial Images Through Score-based Classifications".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

  * `data/`
    * `1`: Folder containing a batch of (960, 3) metrics from sampled images;
    * `2`: Folder containing a batch of (960, 3) metrics from sampled images;
  * `utils/`
    * `data.py`: Methods to aid in extracting desired features from data;
    * `loader.py`: Loads .txt data and saves it in .npy files;
    * `math.py`: Provides mathematical helpers.
    
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

---

## Usage

### Creating the Data

Our first step is to create the data from the available metrics. With that in mind, just run the following script with the input arguments:

```Python
python create_data.py path files -n_samples -normalize -outlier
```

Or, if necessary, invoke the script with its helper:

```Python
python create_data.py -h
```

*Note that it will output a helper file in order to assist in choosing the correct arguments for the script.*

### Digitizing the Data

After creating the `features.npy` file, we want to divide each one of its features into equivalent intervals and discretize their values. In other words, we want to assign a label for each variable concerning each sample. Just choose the following script with the input arguments:

```Python
python create_targets.py input -n_bins
```

### SVM Classification

Finally, after creating the `features.npy` and `targets.npy`, it is now possible to train a classifier and further predict new data. For now, we are using a standard Support Vector Machine classification. Run the following script in order to fulfill that purpose:

```Python
python svm.py features targets
```

### (Optional) Statistical Measures

As an optional procedure, one can also calculate and plot some statistical measures regarding the data. Please use the following scripts in order to accomplish such an approach:

```Python
python make_boxplot.py input
```

```Python
python make_violinplot.py input
```

```Python
python make_histogram.py input
```

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
