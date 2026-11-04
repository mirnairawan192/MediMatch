
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "doc2vec_vectors.csv",
    encoding="utf-8-sig"
)

vector_columns = [
    f"vector_{i+1}"
    for i in range(200)
]

# ==========================================
# 2. AMBIL DATA TESTING
# ==========================================

test_df = df[
    df["split"] == "test"
].copy()

X_test = test_df[vector_columns].values

# ==========================================
# 3. LOAD LABEL ENCODER
# ==========================================

label_encoder = joblib.load(
    "label_encoder.pkl"
)

y_test = label_encoder.transform(
    test_df["specialty"]
)

# ==========================================
# 4. RESHAPE INPUT UNTUK LSTM
# ==========================================

X_test_lstm = X_test.reshape(
    X_test.shape[0],
    1,
    X_test.shape[1]
)

print("Jumlah data testing:", len(test_df))
print("Shape X_test:", X_test_lstm.shape)

# ==========================================
# 5. LOAD MODEL LSTM
# ==========================================

model = tf.keras.models.load_model(
    "lstm_model.keras"
)

print("\nModel LSTM berhasil dimuat.")

# ==========================================
# 6. PREDIKSI DATA TESTING
# ==========================================

y_pred_probability = model.predict(
    X_test_lstm,
    verbose=0
)

y_pred = np.argmax(
    y_pred_probability,
    axis=1
)

# ==========================================
# 7. HITUNG METRIK EVALUASI
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n==========================================")
print("HASIL EVALUASI DATA TESTING")
print("==========================================")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)")

# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm_df)

# Simpan confusion matrix
cm_df.to_csv(
    "confusion_matrix.csv",
    encoding="utf-8-sig"
)

print("\nConfusion matrix disimpan sebagai confusion_matrix.csv")

# ==========================================
# 10. SIMPAN HASIL PREDIKSI
# ==========================================

results_df = test_df[
    ["id", "specialty"]
].copy()

results_df = results_df.rename(
    columns={"specialty": "actual_specialty"}
)

results_df["predicted_specialty"] = (
    label_encoder.inverse_transform(y_pred)
)

results_df["correct_prediction"] = (
    results_df["actual_specialty"]
    == results_df["predicted_specialty"]
)

results_df.to_csv(
    "testing_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nHasil prediksi disimpan sebagai testing_predictions.csv")
print("Evaluasi testing selesai.")
