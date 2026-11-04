
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import joblib

from sklearn.preprocessing import LabelEncoder


# ==========================================
# 1. RANDOM SEED
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. LOAD DATA
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
# 3. SPLIT DATA
# ==========================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "validation"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


# ==========================================
# 4. CREATE X
# ==========================================

X_train = train_df[vector_columns].values
X_val = val_df[vector_columns].values
X_test = test_df[vector_columns].values


# ==========================================
# 5. CREATE Y
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


# ==========================================
# 6. RESHAPE UNTUK LSTM
# ==========================================

X_train_lstm = X_train.reshape(
    X_train.shape[0],
    1,
    X_train.shape[1]
)

X_val_lstm = X_val.reshape(
    X_val.shape[0],
    1,
    X_val.shape[1]
)

X_test_lstm = X_test.reshape(
    X_test.shape[0],
    1,
    X_test.shape[1]
)


# ==========================================
# 7. BUILD MODEL
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(1, 200)
    ),

    tf.keras.layers.LSTM(
        128
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ==========================================
# 8. COMPILE
# ==========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 9. CALLBACKS
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)


# ==========================================
# 10. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 11. TRAINING
# ==========================================

print("\n==========================================")
print("MULAI TRAINING")
print("==========================================")

history = model.fit(
    X_train_lstm,
    y_train,
    validation_data=(
        X_val_lstm,
        y_val
    ),
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ==========================================
# 12. BEST RESULT
# ==========================================

best_epoch_loss = (
    np.argmin(
        history.history["val_loss"]
    ) + 1
)

best_val_loss = min(
    history.history["val_loss"]
)

best_epoch_accuracy = (
    np.argmax(
        history.history["val_accuracy"]
    ) + 1
)

best_val_accuracy = max(
    history.history["val_accuracy"]
)


# ==========================================
# 13. RESULT
# ==========================================

print("\n==========================================")
print("HASIL TRAINING")
print("==========================================")

print(
    "Epoch aktual:",
    len(history.history["loss"])
)

print(
    "Best Epoch Validation Loss:",
    best_epoch_loss
)

print(
    "Best Validation Loss:",
    best_val_loss
)

print(
    "Best Epoch Validation Accuracy:",
    best_epoch_accuracy
)

print(
    "Best Validation Accuracy:",
    best_val_accuracy
)

print(
    "\nBest Validation Accuracy (%):",
    best_val_accuracy * 100
)


# ==========================================
# 14. SAVE MODEL DAN LABEL ENCODER
# ==========================================

model.save(
    "lstm_model.keras"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

print("\n==========================================")
print("PENYIMPANAN MODEL")
print("==========================================")

print(
    "Model berhasil disimpan: lstm_model.keras"
)

print(
    "Label encoder berhasil disimpan: label_encoder.pkl"
)
