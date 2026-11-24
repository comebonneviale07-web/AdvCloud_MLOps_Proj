from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR.parent / "data" / "raw"  # datasets live in data/raw (git-ignored)

data = pd.read_csv(PROJECT_DIR / "ride_features.csv")

X = data.drop(columns=["ride_id", "LABEL"])
y = data["LABEL"]

# Split entire rides, while preserving class proportions as far as possible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Training rides:", len(X_train))
print("Test rides:", len(X_test))
print("\nTest rides per class:")
print(y_test.value_counts().to_string())

models = {
    "Always predict the most common class": DummyClassifier(
        strategy="most_frequent"
    ),
    "Random forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

class_names = sorted(y.unique())

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{'=' * 60}")
    print(name)
    print(
        classification_report(
            y_test,
            predictions,
            labels=class_names,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_test, predictions, labels=class_names)
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=class_names, columns=class_names))
    