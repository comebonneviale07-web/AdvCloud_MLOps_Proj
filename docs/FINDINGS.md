# Data inspection and baseline findings

## Objective and data

The project assignment asks for a model that classifies rides into five categories: Safe, Safe overall, Bad weather, Tandem, and Reckless. The initial analysis uses `GBG500.parquet` and `GBG500_labels.csv`. `SRV3490.parquet` contains crash-related information and has not been used for this five-class baseline.

Although `data.txt` describes the GBG500 labels as evaluation-only and the original idea was unsupervised learning, the teacher confirmed that supervised learning is permitted for this project. The experiments below therefore use labels to train models and evaluate them on separate rides.

The GBG500 parquet file contains 1,529,547 sensor readings from 500 rides. Its columns are `ride_id`, `time_index`, and six sensor channels: `ax`, `ay`, `az`, `rx`, `ry`, and `rz`. The labels file contains one label for each of the 500 rides. All ride IDs match between the two files; there are no duplicate label IDs or missing labels.

| Class | Rides |
| --- | ---: |
| Safe | 416 |
| Safe overall | 38 |
| Bad weather | 22 |
| Tandem | 18 |
| Reckless | 6 |

The classes are heavily imbalanced. Accuracy alone can be misleading: predicting Safe for every ride is correct for most rides but does not identify any minority class.

## Data quality and preprocessing

Rides contain between 1,807 and 4,767 readings. A total of 439 rides contain at least one row with a missing sensor reading. The mean proportion of such rows per ride is approximately 0.65%, and the maximum is approximately 1.58%.

One ride and time-index pair appeared twice, with slightly different sensor values. The preprocessing script averages sensor readings for each duplicate ride/time pair. This is a documented choice; the cause of the duplicate is unknown. The original dataset is not modified.

`prepare_features.py` then computes the mean, standard deviation, minimum, maximum, and median for each of the six sensor channels, producing 30 features per ride. The aggregate statistics ignore individual missing values. The resulting table has 500 rows and no missing feature values. These summary features do not preserve the complete time sequence.

## Initial baseline

`train_baseline.py` uses a stratified 80/20 split by ride, with `random_state=42`. The test set contains 100 rides: 83 Safe, 8 Safe overall, 4 Bad weather, 4 Tandem, and 1 Reckless. It compares a classifier that always predicts the most common class with a random forest using balanced class weights.

| Metric | Always predict Safe | Random forest |
| --- | ---: | ---: |
| Accuracy | 0.83 | 0.80 |
| Macro F1 | 0.18 | 0.36 |
| Reckless recall | 0/1 | 1/1 |
| Bad weather recall | 0/4 | 0/4 |
| Tandem recall | 0/4 | 0/4 |
| Safe overall recall | 0/8 | 2/8 |
| Safe recall | 83/83 | 77/83 |

The random forest improves macro F1 over the most-common-class reference, but it does not identify any Bad weather or Tandem rides in this split. Its successful prediction of the single Reckless test ride is insufficient to establish reliable performance. Scores for all small classes have substantial uncertainty.

## Neural-network experiment

`train_neural_network.py` trains a PyTorch feed-forward network on the same 30 ride-level features and the same stratified 80/20 ride split (`random_state=42`) as the baseline. It has two hidden layers of 64 and 32 units, ReLU activations, and 20% dropout after the first layer. Training uses Adam (learning rate 0.001), a batch size of 32, 100 epochs, and class-weighted cross-entropy. The scaler and class weights are fitted separately on each training partition; validation and test data do not fit the scaler.

Five-fold stratified cross-validation on the 400 training rides produced macro F1 scores of **0.293, 0.346, 0.356, 0.257, and 0.366** (mean **0.324**, standard deviation **0.042**). Each validation fold contains only one Reckless ride, so these scores are unstable. The fixed configuration was then trained on all 400 training rides and evaluated on the 100 test rides:

| Class | Precision | Recall | F1 | Test rides |
| --- | ---: | ---: | ---: | ---: |
| Bad weather | 0.00 | 0.00 | 0.00 | 4 |
| Reckless | 0.00 | 0.00 | 0.00 | 1 |
| Safe | 0.86 | 0.90 | 0.88 | 83 |
| Safe overall | 0.29 | 0.25 | 0.27 | 8 |
| Tandem | 0.00 | 0.00 | 0.00 | 4 |
| **Macro average** | **0.23** | **0.23** | **0.23** | **100** |

Test accuracy was **0.77** and weighted F1 was **0.75**. On this split, the neural network scored below the random forest's 0.36 macro F1 and did not identify the single Reckless test ride. This is one exploratory comparison, not evidence of reliable generalization. The saved `ride_classifier.pt` contains the model weights, feature order, class names, and scaler parameters; it remains a local artifact.

## Next steps and limitations

- Improve and evaluate the deep-learning classifier using training-only cross-validation and per-class metrics.
- Use the existing split as a held-out test set and make model-development decisions using only the training rides. Given just five Reckless rides in the training split, any cross-validation strategy must account for the small class size.
- Because the test results have already been inspected, they are exploratory; reserve a fresh independent evaluation set if one becomes available.
- Confirm whether the assignment's reference to sending an image to the API applies to these trajectory data.
- Track experiments, store model artifacts, and add the API, workflow, tests, containerization, CI/CD, and deployment required by the assignment.

The original datasets and generated feature table are kept local and excluded from Git. Results above describe one split and should not be interpreted as production performance.
