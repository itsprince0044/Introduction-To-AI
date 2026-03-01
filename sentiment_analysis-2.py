# ==========================================================
# SENTIMENT ANALYSIS USING MULTINOMIAL NAÏVE BAYES
# ==========================================================

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# 1️⃣ LOAD DATA
# ==========================================================

df = pd.read_csv("Mobile_Reviews_Sentiment.csv")

print("\n===== DATASET OVERVIEW =====")
print(f"Total Samples: {len(df)}")
print("\nInitial Class Distribution (%):")
print(df["sentiment"].value_counts(normalize=True) * 100)

# ==========================================================
# 2️⃣ TEXT CLEANING
# ==========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["review_text"].apply(clean_text)

print("\nSample Cleaned Text:")
print(df["clean_review"].head(3))

# ==========================================================
# 3️⃣ INJECT 15% LABEL NOISE (BEFORE SPLIT)
# ==========================================================

np.random.seed(42)

noise_fraction = 0.15
n_noise = int(len(df) * noise_fraction)

noise_indices = np.random.choice(df.index, n_noise, replace=False)
labels = df["sentiment"].unique()

for idx in noise_indices:
    current_label = df.loc[idx, "sentiment"]
    new_label = np.random.choice([l for l in labels if l != current_label])
    df.loc[idx, "sentiment"] = new_label

print(f"\nInjected noise into {n_noise} labels (15%)")

print("\nClass Distribution After Noise (%):")
print(df["sentiment"].value_counts(normalize=True) * 100)

# ==========================================================
# 4️⃣ DEFINE FEATURES & TARGET
# ==========================================================

X = df["clean_review"]
y = df["sentiment"]

# ==========================================================
# 5️⃣ TRAIN / TEST SPLIT (70/30 STRATIFIED)
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n===== TRAIN / TEST SPLIT =====")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples : {len(X_test)}")

# ----------------------------------------------------------
# 📊 Visualisation: Train vs Test Size
# ----------------------------------------------------------

plt.figure()
plt.bar(["Training Set", "Test Set"],
        [len(X_train), len(X_test)])
plt.title("70/30 Train-Test Split")
plt.ylabel("Number of Reviews")
plt.show()

# ----------------------------------------------------------
# 📊 Visualisation: Class Distribution
# ----------------------------------------------------------

train_dist = y_train.value_counts(normalize=True) * 100
test_dist = y_test.value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    "Training (%)": train_dist,
    "Testing (%)": test_dist
})

dist_df.plot(kind="bar")
plt.title("Class Distribution: Train vs Test")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.show()

# ==========================================================
# 6️⃣ BUILD NAÏVE BAYES PIPELINE
# ==========================================================

nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=2,
        stop_words="english",
        sublinear_tf=True
    )),
    ("clf", MultinomialNB(alpha=0.5))
])

# ==========================================================
# 7️⃣ TRAIN MODEL
# ==========================================================

nb_pipeline.fit(X_train, y_train)

# ==========================================================
# 8️⃣ TEST SET EVALUATION
# ==========================================================

y_pred = nb_pipeline.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall    = recall_score(y_test, y_pred, average="macro")
f1        = f1_score(y_test, y_pred, average="macro")

print("\n===== TEST SET RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 📊 Visualisation: Model Performance
# ----------------------------------------------------------

metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure()
plt.bar(metrics_names, metrics_values)
plt.title("Naïve Bayes Performance on Test Set")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.show()

# ==========================================================
# 9️⃣ CROSS-VALIDATION (5-FOLD)
# ==========================================================

cv_scores = cross_val_score(
    nb_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\n===== CROSS-VALIDATION =====")
print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ----------------------------------------------------------
# 📊 Visualisation: Cross-Validation
# ----------------------------------------------------------

plt.figure()
plt.bar(range(1, 6), cv_scores)
plt.title("5-Fold Cross-Validation Accuracy")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# ==========================================================
# 🔟 CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y_test, y_pred)
classes = nb_pipeline.named_steps["clf"].classes_

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")

plt.show()

# ==========================================================
# 1️⃣1️⃣ SAMPLE PREDICTIONS
# ==========================================================

samples = [
    "Absolutely love this phone! The camera is fantastic.",
    "Battery drains very fast and performance is bad.",
    "It is average, nothing special."
]

print("\n===== SAMPLE PREDICTIONS =====")
for text in samples:
    prediction = nb_pipeline.predict([text])[0]
    probability = nb_pipeline.predict_proba([text])[0]
    confidence = max(probability)

    print(f"\nReview: {text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")