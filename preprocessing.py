
import pandas as pd
import re
import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data ta fix.csv",
    encoding="utf-8-sig"
)

print("Jumlah data:", len(df))
print("Kolom:", df.columns.tolist())


# ==========================================
# 2. DOWNLOAD NLTK RESOURCES
# ==========================================

nltk.download("punkt_tab")
nltk.download("stopwords")


# ==========================================
# 3. STOPWORDS
# ==========================================

stop_words = set(stopwords.words("english"))


# ==========================================
# 4. CASE FOLDING
# ==========================================

def case_folding(text):
    return str(text).lower()


df["case_folding_text"] = df["patient_text_en"].apply(
    case_folding
)


# ==========================================
# 5. TEXT CLEANING
# ==========================================

def text_cleaning(text):
    text = re.sub(r"[^\w\s]", " ", text)
    return text


df["cleaned_text"] = df["case_folding_text"].apply(
    text_cleaning
)


# ==========================================
# 6. TOKENIZING
# ==========================================

def tokenizing(text):
    return word_tokenize(text)


df["tokenized_text"] = df["cleaned_text"].apply(
    tokenizing
)


# ==========================================
# 7. STOPWORD REMOVAL
# ==========================================

def stopword_removal(tokens):
    return [
        word for word in tokens
        if word not in stop_words
    ]


df["stopword_removed"] = df["tokenized_text"].apply(
    stopword_removal
)


# ==========================================
# 8. COMBINE PREPROCESSING RESULTS
# ==========================================

df["preprocessed_text"] = df["stopword_removed"].apply(
    lambda tokens: " ".join(tokens)
)

print("\nContoh hasil preprocessing:")
print(
    df[
        ["patient_text_en", "preprocessed_text"]
    ].head(5)
)


# ==========================================
# 9. SPLIT DATA: 80:10:10
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["specialty"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["specialty"]
)

print("\nPembagian data:")
print("Training  :", len(train_df))
print("Validation:", len(val_df))
print("Testing   :", len(test_df))


# ==========================================
# 10. PREPARE TRAINING DOCUMENTS
# ==========================================

tagged_documents = [
    TaggedDocument(
        words=text.split(),
        tags=[i]
    )
    for i, text in enumerate(
        train_df["preprocessed_text"]
    )
]


# ==========================================
# 11. DOC2VEC EXPERIMENT CONFIGURATION
# ==========================================

# Eksperimen pertama:
VECTOR_SIZE = 150

# Pertahankan parameter lainnya.
D2V_EPOCHS = 40
INFER_EPOCHS = 40
WINDOW_SIZE = 5
MIN_COUNT = 1
DM = 0

print("\nKonfigurasi Doc2Vec:")
print("Vector size:", VECTOR_SIZE)
print("Window     :", WINDOW_SIZE)
print("Min count  :", MIN_COUNT)
print("Epochs     :", D2V_EPOCHS)
print("DM         :", DM)


# ==========================================
# 12. BUILD DOC2VEC MODEL
# ==========================================

model_d2v = Doc2Vec(
    vector_size=VECTOR_SIZE,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=1,
    epochs=D2V_EPOCHS,
    dm=DM,
    seed=42
)

model_d2v.build_vocab(tagged_documents)


# ==========================================
# 13. TRAIN DOC2VEC
# ==========================================

model_d2v.train(
    tagged_documents,
    total_examples=model_d2v.corpus_count,
    epochs=D2V_EPOCHS
)

model_file = f"doc2vec_model_{VECTOR_SIZE}_dm{DM}.model"

model_d2v.save(model_file)

print("\nModel Doc2Vec disimpan:", model_file)


# ==========================================
# 14. CREATE TRAINING VECTORS
# ==========================================

X_train_text = np.array([
    model_d2v.dv[i]
    for i in range(len(train_df))
])


# ==========================================
# 15. CREATE VALIDATION VECTORS
# ==========================================

X_val_text = np.array([
    model_d2v.infer_vector(
        str(text).split(),
        epochs=INFER_EPOCHS
    )
    for text in val_df["preprocessed_text"]
])


# ==========================================
# 16. CREATE TESTING VECTORS
# ==========================================

X_test_text = np.array([
    model_d2v.infer_vector(
        str(text).split(),
        epochs=INFER_EPOCHS
    )
    for text in test_df["preprocessed_text"]
])


print("\nBentuk vektor:")
print("X_train_text:", X_train_text.shape)
print("X_val_text  :", X_val_text.shape)
print("X_test_text :", X_test_text.shape)


# ==========================================
# 17. SAVE PREPROCESSED DATA
# ==========================================

df.to_csv(
    "data_preprocessed.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nData preprocessing disimpan: data_preprocessed.csv")


# ==========================================
# 18. CREATE VECTOR DATAFRAMES
# ==========================================

vector_columns = [
    f"vector_{i+1}"
    for i in range(VECTOR_SIZE)
]


def create_vector_dataframe(
    vectors,
    source_df,
    split_name
):
    result = pd.DataFrame(
        vectors,
        columns=vector_columns
    )

    result.insert(
        0,
        "id",
        source_df["id"].values
    )

    result["split"] = split_name
    result["specialty"] = source_df["specialty"].values

    return result


train_vectors_df = create_vector_dataframe(
    X_train_text,
    train_df,
    "train"
)

val_vectors_df = create_vector_dataframe(
    X_val_text,
    val_df,
    "validation"
)

test_vectors_df = create_vector_dataframe(
    X_test_text,
    test_df,
    "test"
)


# ==========================================
# 19. COMBINE AND SAVE DOC2VEC VECTORS
# ==========================================

doc2vec_vectors_df = pd.concat(
    [
        train_vectors_df,
        val_vectors_df,
        test_vectors_df
    ],
    ignore_index=True
)

vectors_file = f"doc2vec_vectors_{VECTOR_SIZE}_dm{DM}.csv"

doc2vec_vectors_df.to_csv(
    vectors_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nDoc2Vec vectors disimpan:", vectors_file)
print("Shape Doc2Vec vectors:", doc2vec_vectors_df.shape)


# ==========================================
# 20. ENCODE SPECIALTY LABELS
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
# 21. DISPLAY SPECIALTY MAPPING
# ==========================================

print("\nMapping Specialty:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)


# ==========================================
# 22. CHECK LABEL SHAPES
# ==========================================

print("\nShape label:")
print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

print(
    "\nJumlah kelas specialty:",
    len(label_encoder.classes_)
)

print("\nEksperimen Doc2Vec selesai!")
