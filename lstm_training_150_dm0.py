
import os
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. LOAD DOC2VEC VECTORS
# ==========================================

df = pd.read_csv(
    "doc2vec_vectors_150_dm0.csv",
    encoding="utf-8-sig"
)

print("Jumlah data:", len(df))
print("Kolom:", df.columns.tolist())


# ==========================================
# 3. SEPARATE DATA BY SPLIT
# ==========================================

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nPembagian data:")
print("Training  :", len(train_df))
print("Validation:", len(val_df))
print("Testing   :", len(test_df))


# ==========================================
# 4. PREPARE FEATURE COLUMNS
# ==========================================

vector_columns = [
    col for col in df.columns
    if col.startswith("vector_")
]

vector_columns = sorted(
    vector_columns,
    key=lambda col: int(col.split("_")[1])
)

if len(vector_columns) != 150:
    raise ValueError(
        f"Seharusnya ada 150 fitur, "
        f"tetapi ditemukan {len(vector_columns)}."
    )

X_train = train_df[vector_columns].to_numpy(
    dtype=np.float32
)

X_val = val_df[vector_columns].to_numpy(
    dtype=np.float32
)

X_test = test_df[vector_columns].to_numpy(
    dtype=np.float32
)


# ==========================================
# 5. ENCODE SPECIALTY LABELS
# ==========================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    train_df["specialty"]
)

y_val = label_encoder.transform(
    val_df["specialty"]
)

y_test = label_encoder.transform(
    test_df["specialty"]
)

num_classes = len(label_encoder.classes_)

print("\nMapping specialty:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

print("\nJumlah kelas:", num_classes)


# ==========================================
# 6. RESHAPE INPUT FOR LSTM
# ==========================================

X_train = X_train.reshape(
    X_train.shape[0], 1, X_train.shape[1]
)

X_val = X_val.reshape(
    X_val.shape[0], 1, X_val.shape[1]
)

X_test = X_test.reshape(
    X_test.shape[0], 1, X_test.shape[1]
)

print("\nShape input LSTM:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)


# ==========================================
# 7. BUILD LSTM MODEL
# ==========================================

model = Sequential([
    LSTM(
        128,
        input_shape=(1, 150)
    ),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 8. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1
)


# ==========================================
# 9. TRAIN LSTM
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ==========================================
# 10. SAVE MODEL AND LABEL ENCODER
# ==========================================

model.save("lstm_model_150_dm0.keras")

with open("label_encoder_150_dm0.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel disimpan:")
print("lstm_model_150_dm0.keras")
print("label_encoder_150_dm0.pkl")


# ==========================================
# 11. VALIDATION EVALUATION
# ==========================================

val_loss, val_accuracy = model.evaluate(
    X_val, y_val, verbose=0
)

y_val_pred = np.argmax(
    model.predict(X_val, verbose=0),
    axis=1
)

val_precision = precision_score(
    y_val, y_val_pred,
    average="macro",
    zero_division=0
)

val_recall = recall_score(
    y_val, y_val_pred,
    average="macro",
    zero_division=0
)

val_f1 = f1_score(
    y_val, y_val_pred,
    average="macro",
    zero_division=0
)

print("\n========== VALIDATION RESULTS ==========")
print(f"Validation loss      : {val_loss:.4f}")
print(f"Validation accuracy  : {val_accuracy * 100:.2f}%")
print(f"Validation precision : {val_precision * 100:.2f}%")
print(f"Validation recall    : {val_recall * 100:.2f}%")
print(f"Validation macro F1   : {val_f1 * 100:.2f}%")


# ==========================================
# 12. TEST EVALUATION
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

test_probabilities = model.predict(
    X_test, verbose=0
)

y_test_pred = np.argmax(
    test_probabilities,
    axis=1
)

test_precision = precision_score(
    y_test, y_test_pred,
    average="macro",
    zero_division=0
)

test_recall = recall_score(
    y_test, y_test_pred,
    average="macro",
    zero_division=0
)

test_f1 = f1_score(
    y_test, y_test_pred,
    average="macro",
    zero_division=0
)

print("\n========== TEST RESULTS ==========")
print(f"Test loss      : {test_loss:.4f}")
print(f"Test accuracy  : {test_accuracy * 100:.2f}%")
print(f"Test precision : {test_precision * 100:.2f}%")
print(f"Test recall    : {test_recall * 100:.2f}%")
print(f"Test macro F1  : {test_f1 * 100:.2f}%")


# ==========================================
# 13. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        labels=np.arange(num_classes),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# 14. SAVE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=np.arange(num_classes)
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_df.to_csv(
    "confusion_matrix_150_dm0.csv",
    encoding="utf-8-sig"
)


# ==========================================
# 15. SAVE TEST PREDICTIONS
# ==========================================

test_predictions_df = pd.DataFrame({
    "id": test_df["id"].values,
    "actual_specialty": label_encoder.inverse_transform(y_test),
    "predicted_specialty": label_encoder.inverse_transform(y_test_pred),
    "correct": y_test == y_test_pred,
    "confidence": np.max(test_probabilities, axis=1)
})

test_predictions_df.to_csv(
    "testing_predictions_150_dm0.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 16. SAVE EXPERIMENT SUMMARY
# ==========================================

summary_df = pd.DataFrame([{
    "experiment": "Doc2Vec_150_dm0",
    "vector_size": 150,
    "dm": 0,
    "window": 5,
    "min_count": 1,
    "doc2vec_epochs": 40,
    "lstm_epochs_run": len(history.history["loss"]),
    "validation_accuracy": val_accuracy,
    "validation_macro_precision": val_precision,
    "validation_macro_recall": val_recall,
    "validation_macro_f1": val_f1,
    "test_accuracy": test_accuracy,
    "test_macro_precision": test_precision,
    "test_macro_recall": test_recall,
    "test_macro_f1": test_f1
}])

summary_df.to_csv(
    "experiment_summary_150_dm0.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSemua evaluasi selesai.")
print("Confusion matrix: confusion_matrix_150_dm0.csv")
print("Test predictions: testing_predictions_150_dm0.csv")
print("Ringkasan eksperimen: experiment_summary_150_dm0.csv")
