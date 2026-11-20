from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parent
SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RideClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(X_train, y_train, n_classes, seed):
    set_seed(seed)

    # Fit the scaler only on this training partition.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = RideClassifier(X_train.shape[1], n_classes)

    # Give underrepresented classes more weight in the loss.
    counts = np.bincount(y_train, minlength=n_classes)
    weights = len(y_train) / (n_classes * counts)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for _ in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    return model, scaler


def predict(model, scaler, X):
    X_scaled = scaler.transform(X)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def main():
    set_seed(SEED)

    data = pd.read_csv(PROJECT_DIR / "ride_features.csv")
    feature_names = data.drop(columns=["ride_id", "LABEL"]).columns.tolist()
    X = data[feature_names].to_numpy(dtype=np.float32)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data["LABEL"])
    n_classes = len(encoder.classes_)

    # Same ride-level split and seed as the random-forest baseline.
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Evaluate the fixed model configuration on the training set.
    # Five folds are possible because training contains five reckless rides.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        model, scaler = train_model(
            X_train[fit_idx], y_train[fit_idx], n_classes, SEED + fold
        )
        predictions = predict(model, scaler, X_train[val_idx])
        score = f1_score(
            y_train[val_idx],
            predictions,
            labels=np.arange(n_classes),
            average="macro",
            zero_division=0,
        )
        fold_scores.append(score)
        print(f"Fold {fold} macro F1: {score:.3f}")

    print(f"\nMean training cross-validation macro F1: {np.mean(fold_scores):.3f}")
    print(f"Fold standard deviation: {np.std(fold_scores):.3f}")

    # Train the fixed configuration on all training rides.
    model, scaler = train_model(X_train, y_train, n_classes, SEED)
    predictions = predict(model, scaler, X_test)

    print("\nHeld-out test results (exploratory):")
    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(n_classes),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=np.arange(n_classes))
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=encoder.classes_, columns=encoder.classes_))

    # Save everything needed to apply the same model to new feature rows.
    artifact = {
        "model_state": copy.deepcopy(model.state_dict()),
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "feature_names": feature_names,
        "classes": encoder.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    artifact_path = PROJECT_DIR / "ride_classifier.pt"
    torch.save(artifact, artifact_path)
    print(f"\nSaved model artifact: {artifact_path}")


if __name__ == "__main__":
    main()