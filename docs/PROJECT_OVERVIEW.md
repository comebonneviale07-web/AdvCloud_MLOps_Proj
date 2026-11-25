# Project overview and file guide

> **Update (repository restructured):** the exploration scripts below now live in `research/` and read
> the datasets from `data/raw/`. The production pipeline (MLflow, API, Docker, Kubernetes, CI/CD,
> monitoring) is in `src/trajclf/` and described in the root `README.md`; the AI-infrastructure
> contribution is in `ai-infrastructure/`. The assignment's workshop guide moved to
> `ai-infrastructure/ASSIGNMENT.md`.

## Purpose

This project builds a reproducible machine-learning pipeline to classify ride trajectories into five categories: Safe, Safe overall, Bad weather, Tandem, and Reckless. It uses sensor readings from the GBG500 dataset and explores how well different models distinguish these classes. The assignment also calls for experiment tracking, model storage, a prediction API, an interactive app, tests, a workflow, Docker, Kubernetes, and CI/CD. These components are now implemented in `src/trajclf/` (see the root README); this file documents the initial research phase.

The description in `data.txt` presents an anomaly-detection objective and says the GBG500 labels are for evaluation. The teacher subsequently confirmed that, although the original idea was unsupervised learning, supervised learning is permitted for this project. The current experiments therefore use the GBG500 labels for training and evaluation on separate rides.

## Python scripts

| File | Purpose |
| --- | --- |
| `inspect_data.py` | Inspects the labels and parquet schemas, sizes, sample rows, and missing values. Parquet sample statistics use the first row group, not the full file. |
| `check_rides.py` | Checks ride IDs against the labels, summarizes ride lengths and missing sensor data, and counts duplicate ride/time pairs across GBG500. |
| `prepare_features.py` | Averages duplicate ride/time readings, calculates five statistics for each of six sensor channels, attaches labels, and writes `ride_features.csv`. |
| `train_baseline.py` | Splits rides into training and test sets and compares a most-common-class predictor with a random forest, reporting per-class metrics and confusion matrices. |
| `train_neural_network.py` | Trains a PyTorch classifier on ride-level features, evaluates it with training-set cross-validation and the exploratory test set, and saves a model artifact with scaling information. |

## Text files and generated outputs

| File | Purpose |
| --- | --- |
| `data.txt` | Describes the supplied datasets and their labels. |
| `data_inspection.txt` | Saved console output from `inspect_data.py`. |
| `ride_check.txt` | Saved output from `check_rides.py`. |
| `feature_check.txt` | Saved output from feature preparation, including class counts and missing feature values. |
| `baseline_results.txt` | Saved baseline classification reports and confusion matrices. |
| `neural_network_results.txt` | Saved cross-validation scores and the neural network's exploratory test results. |

The output text files are snapshots of individual runs. Regenerate them after changing a script or its inputs; check `.gitignore` to see which are excluded from version control. The most relevant findings and experimental conditions are summarized in `FINDINGS.md`.

## Other important files

| File | Purpose |
| --- | --- |
| `GBG500.parquet` and `GBG500_labels.csv` | Local sensor readings and five-class labels used by the current experiments. |
| `SRV3490.parquet` | Local crash-related dataset; not used in the current five-class experiments. |
| `ride_features.csv` | Generated ride-level features and labels; kept local because it is derived from the dataset. |
| `ride_classifier.pt` | Generated neural-network weights and preprocessing metadata; currently stored locally. |
| `FINDINGS.md` | Data-quality findings, experiment conditions, metrics, and limitations. |
| `AdvCloud_README.md` | Additional AI infrastructure workshop guidance, including benchmarking and scaling. |
| `.gitignore` | Lists files and directories excluded from Git, including the local datasets. |

## Current workflow

With the datasets present locally and the Python environment activated, run `inspect_data.py` and `check_rides.py` to inspect the data, then `prepare_features.py` to generate the feature table. Run `train_baseline.py` and `train_neural_network.py` to reproduce the initial experiments. These scripts are currently research prototypes; model tracking, containerization, serving, and deployment still need to be implemented.
