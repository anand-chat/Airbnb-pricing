# Airbnb-pricing

## DESCRIPTION

Code for training and evaluating an Airbnb price prediction model. Made by Team Data Viz-ards for CSE 6242 Project, Spring 2020.

## INSTALLATION

Ensure that you have [Python 3.7.6](https://www.python.org/downloads/release/python-376/) and [`virtualenv`](https://virtualenv.pypa.io/en/latest/installation.html#installation) installed. Then perform the following steps:

1. Clone or download this repository and `cd` into it.
2. Create a Python 3.7.6 `virtualenv`, and activate it:

```shell
virtualenv --python=path_to_your_python_executable venv
source venv/bin/activate
```

3. Install dependencies using `pip`:

```shell
pip install -r requirements.txt
```

## EXECUTION

Simply run:

```
python AirbnbModel.py
```

This will train and evaluate the model on the provided dataset - `updatedFinalData.csv` and generate output files to store predictions, feature importance values, and save the model to disk.
