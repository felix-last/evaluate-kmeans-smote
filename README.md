# Evaluate KMeans-SMOTE
This project is used to evaluate the performance of the oversampling method KMeans-SMOTE.

## Dependencies

1. Install imbalanced-learn from https://github.com/felix-last/imbalanced-learn.
2. Install imbalanced-tools from https://github.com/felix-last/imbalanced-tools.
3. `pip3 install -r requirements.txt`
3. (Optional) Use [vscode's python extension](https://github.com/DonJayamanne/pythonVSCode) for executing python script in Jupyter notebook.
4. (Optional) Use https://github.com/gatsoulis/py2ipynb to convert the python scripts to iPython Notebook format.

## Usage
1. Set local folder paths using `config.yml` (see [`config.sample.yml`](config.sample.yml) for an example).
2. Open `imbalanced_benchmark.py` to check and adapt `experiment_config` and `oversampling_methods`.
3. Execute `imbalanced_benchmark.py`.
4. Once the script has run, open the results folder created with the current timestamp. It contains a PDF with plots and CSV files for all statistics provided by [imbalanced-tools](https://github.com/felix-last/imbalanced-tools).
