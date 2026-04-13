import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

base_path = "."
window_size = 20
step = 5

X, y = [], []

# -------------------------------
# LOAD ALL SUBJECTS
# -------------------------------
for folder in os.listdir(base_path):

    if not folder.startswith("S"):
        continue

    file1 = os.path.join(base_path, folder, f"{folder}.pkl")
    file2 = os.path.join(base_path, folder, folder, f"{folder}.pkl")

    if os.path.exists(file1):
        file_path = file1
    elif os.path.exists(file2):
        file_path = file2
    else:
        continue

    print(f"Loading {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    eda = data['signal']['wrist']['EDA'].reshape(-1)
    labels = data['label'].reshape(-1)

    # -------------------------------
    # FIX LABEL LENGTH MISMATCH
    # -------------------------------
    if len(labels) != len(eda):
        factor = len(labels) / len(eda)
        labels = np.array([labels[int(i * factor)] for i in range(len(eda))])

    # Normalize
    eda = (eda - np.mean(eda)) / np.std(eda)

    # -------------------------------
    # WINDOWING (CENTER LABEL)
    # -------------------------------
    for i in range(0, len(eda) - window_size, step):

        window = eda[i:i + window_size]
        label = labels[i + window_size // 2]

        if label in [1, 2, 3]:
            X.append(window)
            y.append(label)

# Convert
X = np.array(X)
y = np.array(y)

print("\nTotal samples:", X.shape)

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# -------------------------------
# TRAIN
# -------------------------------
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# TEST
# -------------------------------
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print("\nFinal Accuracy:", accuracy)

# -------------------------------
# EXTRA VALIDATION (IMPORTANT)
# -------------------------------
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
with open("stress_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as stress_model.pkl")