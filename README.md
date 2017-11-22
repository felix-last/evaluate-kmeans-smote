# Evaluate K-Means SMOTE
This project is used to evaluate the performance of the oversampling method [k-means SMOTE](https://github.com/felix-last/kmeans_smote).

## Dependencies

1. `pip3 install -r requirements.txt`

## Usage
1. Set local folder paths using `config.yml` (see [`config.sample.yml`](config.sample.yml) for an example).
2. Open `imbalanced_benchmark.py` to check and adapt `experiment_config`, `classifiers` and `oversampling_methods`.
3. Execute `imbalanced_benchmark.py`.
4. Once the script has run, open the results folder created with the current timestamp. It contains a pickled experiment (or CSV files for older versions of imbalanced-tools) which can be read by [imbalanced-tools](https://github.com/felix-last/imbalanced-tools).
5. Execute `plot_imbalanced_benchmark.py` with the timestamp as a parameter. This will generate a PDF file with plots in the experiment's folder.
