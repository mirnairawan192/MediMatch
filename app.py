
import os
import re
import pickle
from functools import lru_cache

# =========================================================
# 1. ENVIRONMENT CONFIGURATION
# =========================================================

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
from transformers import MarianTokenizer, MarianMTModel

torch.set_num_threads(1)

import numpy as np
import pandas as pd
import nltk

from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.models import load_model

from langdetect import detect, LangDetectException, DetectorFactory


# =========================================================
# 2. CONFIGURATION
# =========================================================

DetectorFactory.seed = 0

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC2VEC_PATH = os.path.join(
    BASE_DIR, "doc2vec_model_150_dm0.model"
)

LSTM_PATH = os.path.join(
    BASE_DIR, "lstm_model_150_dm0.keras"
)

ENCODER_PATH = os.path.join(
    BASE_DIR, "label_encoder_150_dm0.pkl"
)

MAPPING_PATH = os.path.join(
    BASE_DIR, "mapping_speciality.csv"
)

DOCTORS_PATH = os.path.join(
    BASE_DIR, "jadwal_dokter_stella_maris_2025.csv"
)

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-id-en"

translator_tokenizer = None
translator_model = None


# =========================================================
# 3. CHECK REQUIRED FILES
# =========================================================

required_files = [
    DOC2VEC_PATH,
    LSTM_PATH,
    ENCODER_PATH,
    MAPPING_PATH,
    DOCTORS_PATH
]

for file_path in required_files:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {os.path.basename(file_path)}"
        )


# =========================================================
# 4. LOAD DOC2VEC, LSTM, AND LABEL ENCODER
# =========================================================

print("Memuat model Doc2Vec...")
doc2vec_model = Doc2Vec.load(DOC2VEC_PATH)

print("Memuat model LSTM...")
lstm_model = load_model(LSTM_PATH)

print("Memuat Label Encoder...")

with open(ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

if doc2vec_model.vector_size != 150:
    raise ValueError("Ukuran vektor Doc2Vec harus 150.")

if lstm_model.input_shape[-1] != 150:
    raise ValueError("Ukuran input LSTM tidak sesuai.")

if lstm_model.output_shape[-1] != len(label_encoder.classes_):
    raise ValueError(
        "Jumlah output LSTM tidak sesuai dengan Label Encoder."
    )

print("Model berhasil dimuat.")
print("Jumlah kelas:", len(label_encoder.classes_))
print("Daftar kelas:", list(label_encoder.classes_))


# =========================================================
# 5. LOAD SPECIALTY MAPPING
# =========================================================

mapping_df = pd.read_csv(
    MAPPING_PATH,
    encoding="utf-8-sig"
)

mapping_df.columns = (
    mapping_df.columns
    .astype(str)
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

required_mapping_columns = {
    "kaggle_specialty",
    "rs_poliklinik"
}

if not required_mapping_columns.issubset(mapping_df.columns):
    raise ValueError(
        "Kolom mapping_speciality.csv harus memiliki "
        "kaggle_specialty dan rs_poliklinik."
    )

mapping_df = mapping_df.fillna("")

specialty_mapping = {
    str(row["kaggle_specialty"]).strip().lower():
    str(row["rs_poliklinik"]).strip()
    for _, row in mapping_df.iterrows()
    if str(row["kaggle_specialty"]).strip()
}

print("Mapping spesialisasi berhasil dimuat.")


# =========================================================
# 6. LOAD DOCTOR DATABASE
# =========================================================

doctors_df = pd.read_csv(
    DOCTORS_PATH,
    encoding="utf-8-sig"
)

doctors_df.columns = (
    doctors_df.columns
    .astype(str)
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

required_doctor_columns = {
    "poliklinik",
    "nama_dokter"
}

if not required_doctor_columns.issubset(doctors_df.columns):
    raise ValueError(
        "Kolom database dokter harus memiliki "
        "poliklinik dan nama_dokter."
    )

doctors_df = doctors_df.fillna("")

doctors_df["poliklinik"] = (
    doctors_df["poliklinik"].astype(str).str.strip()
)

doctors_df["nama_dokter"] = (
    doctors_df["nama_dokter"].astype(str).str.strip()
)

doctors_df = doctors_df[
    (doctors_df["poliklinik"] != "")
    & (doctors_df["nama_dokter"] != "")
]


def normalize_polyclinic(name):
    return re.sub(
        r"\s+",
        " ",
        str(name).strip().upper()
    )


def get_doctors_by_polyclinic(polyclinic):

    target = normalize_polyclinic(polyclinic)

    filtered = doctors_df[
        doctors_df["poliklinik"].apply(normalize_polyclinic)
        == target
    ]

    return filtered["nama_dokter"].drop_duplicates().tolist()


# =========================================================
# 7. LOAD NLTK RESOURCES
# =========================================================

for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as error:
        print("NLTK resource warning:", error)

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = set()


# =========================================================
# 8. TRANSLATION MODEL
# =========================================================

def load_translation_model():

    global translator_tokenizer
    global translator_model

    if translator_model is None:

        print("Memuat model penerjemah...")

        translator_tokenizer = MarianTokenizer.from_pretrained(
            TRANSLATION_MODEL
        )

        translator_model = MarianMTModel.from_pretrained(
            TRANSLATION_MODEL
        )

        translator_model.eval()

        print("Model penerjemah berhasil dimuat.")

    return translator_tokenizer, translator_model


def detect_complaint_language(text):

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"


@lru_cache(maxsize=128)
def translate_to_english(text):

    text = str(text).strip()

    if not text:
        raise ValueError("Keluhan kosong.")

    language = detect_complaint_language(text)

    print("Bahasa terdeteksi:", language)

    if language == "en":
        return text

    tokenizer, model = load_translation_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            num_beams=2,
            max_new_tokens=64,
            early_stopping=True
        )

    translated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if not translated_text:
        raise ValueError(
            f"Hasil terjemahan kosong untuk keluhan: {text}"
        )

    return translated_text


# =========================================================
# 9. TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = re.sub(
        r"[^\w\s]",
        " ",
        str(text).lower()
    )

    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


# =========================================================
# 10. SPLIT MULTIPLE COMPLAINTS
# =========================================================

def split_complaints(complaint):

    complaint = str(complaint).strip()

    if not complaint:
        return []

    # Pisahkan berdasarkan baris baru dan titik koma.
    complaint = re.sub(
        r"[\r\n;]+",
        "|",
        complaint
    )

    # Pisahkan berdasarkan tanda baca.
    complaint = re.sub(
        r"[.!?]+\s*",
        "|",
        complaint
    )

    # Pisahkan berdasarkan kata penghubung.
    conjunction_pattern = (
        r"\s+(?:dan|serta|kemudian|sedangkan|"
        r"and|also|but|while|whereas)\s+"
    )

    parts = re.split(
        conjunction_pattern,
        complaint,
        flags=re.IGNORECASE
    )

    valid_parts = []

    for part in parts:

        part = part.strip(" ,.-")

        if part and preprocess_text(part):
            valid_parts.append(part)

    # Hapus keluhan yang benar-benar identik.
    cleaned_parts = []
    seen = set()

    for part in valid_parts:

        key = re.sub(
            r"\s+",
            " ",
            part.lower()
        )

        if key not in seen:
            seen.add(key)
            cleaned_parts.append(part)

    print("Hasil pemisahan keluhan:", cleaned_parts)

    return cleaned_parts if cleaned_parts else [complaint]


# =========================================================
# 11. SPECIALTY OVERRIDE
# =========================================================

def detect_specialty_override(complaint):

    complaint = str(complaint).lower()

    # -----------------------------------------------------
    # PREGNANCY
    # -----------------------------------------------------

    pregnancy_keywords = [
        r"\bhamil\b",
        r"\bkehamilan\b",
        r"\bmengandung\b",
        r"\bpregnant\b",
        r"\bpregnancy\b"
    ]

    if any(
        re.search(pattern, complaint)
        for pattern in pregnancy_keywords
    ):

        gynecology_polyclinic = specialty_mapping.get(
            "gynecology",
            ""
        )

        if gynecology_polyclinic:

            return {
                "specialty": "gynecology",
                "poliklinik": gynecology_polyclinic
            }

    # -----------------------------------------------------
    # EYE
    # -----------------------------------------------------

    eye_keywords = [
        r"\bmata\b",
        r"\bkelopak mata\b",
        r"\bbola mata\b",
        r"\bsakit mata\b",
        r"\bnyeri mata\b",
        r"\bmata merah\b",
        r"\bmata perih\b",
        r"\bmata gatal\b",
        r"\bmata berair\b",
        r"\bpenglihatan\b",
        r"\bpandangan kabur\b",
        r"\bpenglihatan buram\b",
        r"\bblurred vision\b",
        r"\bblurry vision\b",
        r"\bvision loss\b",
        r"\beye pain\b",
        r"\bsore eyes\b",
        r"\bred eye\b",
        r"\bred eyes\b",
        r"\bitchy eyes\b",
        r"\bwatery eyes\b",
        r"\beyelid\b",
        r"\beyes?\b"
    ]

    # -----------------------------------------------------
    # DENTAL
    # -----------------------------------------------------

    dental_keywords = [
        r"\bgigi\b",
        r"\bgusi\b",
        r"\bgeraham\b",
        r"\bsakit gigi\b",
        r"\bnyeri gigi\b",
        r"\bgigi berlubang\b",
        r"\bgigi patah\b",
        r"\bgigi copot\b",
        r"\bgigi tumbuh\b",
        r"\bgusi bengkak\b",
        r"\bgusi berdarah\b",
        r"\btoothache\b",
        r"\btooth pain\b",
        r"\bdental pain\b",
        r"\bsore gums\b",
        r"\bswollen gums\b",
        r"\bbleeding gums\b",
        r"\bcavity\b",
        r"\bteeth\b",
        r"\btooth\b"
    ]

    if any(
        re.search(pattern, complaint)
        for pattern in dental_keywords
    ):

        return {
            "specialty": "dentistry",
            "poliklinik": "GIGI"
        }

    if any(
        re.search(pattern, complaint)
        for pattern in eye_keywords
    ):

        return {
            "specialty": "ophthalmology",
            "poliklinik": "MATA"
        }

    # Tidak ada aturan otomatis untuk demam atau batuk.
    # Keluhan tersebut tetap diproses oleh Doc2Vec + LSTM.

    return None


# =========================================================
# 12. PREDICT ONE COMPLAINT USING DOC2VEC + LSTM
# =========================================================

def predict_one_complaint(complaint, translated_complaint):

    tokens = preprocess_text(translated_complaint)

    print("Token hasil preprocessing:", tokens)

    if not tokens:

        raise ValueError(
            "Tidak ada kata yang dapat diproses dari hasil "
            "terjemahan: " + translated_complaint
        )

    # DOC2VEC
    vector = doc2vec_model.infer_vector(
        tokens,
        epochs=40
    )

    vector = np.asarray(
        vector,
        dtype=np.float32
    ).reshape(1, 1, 150)

    # LSTM
    probabilities = lstm_model.predict(
        vector,
        verbose=0
    )[0]

    if len(probabilities) != len(label_encoder.classes_):

        raise ValueError(
            "Output LSTM tidak sesuai dengan Label Encoder."
        )

    # Urutkan tiga kelas dengan probabilitas tertinggi.
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = []

    for class_index in top_indices:

        top_predictions.append({
            "specialty": str(
                label_encoder.classes_[class_index]
            ).strip().lower(),

            "confidence": float(
                probabilities[class_index] * 100
            )
        })

    predicted_index = int(np.argmax(probabilities))

    specialty = str(
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    ).strip().lower()

    confidence = float(
        probabilities[predicted_index] * 100
    )

    print("\n========== LSTM PREDICTION ==========")
    print("Keluhan:", complaint)
    print("Terjemahan:", translated_complaint)
    print("Top 3 prediksi:")

    for item in top_predictions:

        print(
            item["specialty"],
            ":",
            f'{item["confidence"]:.2f}%'
        )

    print("Prediksi utama:", specialty)
    print("Confidence:", f"{confidence:.2f}%")

    return {
        "specialty": specialty,
        "confidence": confidence,
        "complaint": complaint,
        "translated_complaint": translated_complaint,
        "top_predictions": top_predictions
    }


# =========================================================
# 13. MAP SPECIALTY TO POLYCLINIC
# =========================================================

def map_recommendation(recommendation):

    specialty = recommendation["specialty"]

    poliklinik = specialty_mapping.get(
        specialty,
        ""
    )

    if (
        not poliklinik
        or poliklinik.strip().upper() == "PENDING"
    ):

        return {
            **recommendation,
            "poliklinik": None,
            "status": "unmapped"
        }

    return {
        **recommendation,
        "poliklinik": poliklinik,
        "status": "normal"
    }


# =========================================================
# 14. PROCESS EACH COMPLAINT INDEPENDENTLY
# =========================================================

def process_all_complaints(complaint):

    complaint_parts = split_complaints(complaint)

    recommendations = []
    emergency_predictions = []
    translation_results = []

    print("\n========== MULTI COMPLAINT PROCESSING ==========")
    print("Jumlah keluhan:", len(complaint_parts))

    for index, part in enumerate(complaint_parts, start=1):

        print(f"\nKeluhan {index}: {part}")

        # Terjemahkan setiap keluhan secara terpisah.
        translated = translate_to_english(part)

        translation_results.append(translated)

        print("Hasil terjemahan:", translated)

        # Periksa aturan khusus.
        override = (
            detect_specialty_override(part)
            or detect_specialty_override(translated)
        )

        if override:

            recommendation = {
                "specialty": override["specialty"],
                "confidence": None,
                "complaint": part,
                "translated_complaint": translated,
                "poliklinik": override["poliklinik"],
                "status": "normal",
                "method": "rule"
            }

            print(
                "Prediksi menggunakan aturan khusus:",
                recommendation["specialty"]
            )

        else:

            prediction = predict_one_complaint(
                part,
                translated
            )

            recommendation = map_recommendation(
                prediction
            )

            recommendation["method"] = "Doc2Vec + LSTM"

        # Emergency tetap ditampilkan sebagai kartu tersendiri.
        if recommendation["specialty"] == "emergency":

            recommendation["status"] = "emergency"
            recommendation["poliklinik"] = None

            emergency_predictions.append(recommendation)

        recommendations.append(recommendation)

        print(
            "Hasil akhir:",
            recommendation["specialty"],
            "| Poliklinik:",
            recommendation.get("poliklinik"),
            "| Confidence:",
            recommendation.get("confidence"),
            "| Method:",
            recommendation.get("method")
        )

    if not recommendations:

        result_type = "error"

    elif all(
        item["status"] == "emergency"
        for item in recommendations
    ):

        result_type = "emergency"

    elif any(
        item["status"] == "unmapped"
        for item in recommendations
    ):

        result_type = "unmapped"

    else:

        result_type = "normal"

    return {
        "type": result_type,
        "recommendations": recommendations,
        "emergency_detected": bool(emergency_predictions),
        "emergency_predictions": emergency_predictions,
        "translation_results": translation_results
    }


# =========================================================
# 15. MAIN PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":

        return render_template("index.html")

    complaint = request.form.get(
        "complaint",
        ""
    ).strip()

    if not complaint:

        return render_template(
            "result.html",
            result=None,
            error="Silakan masukkan keluhan terlebih dahulu.",
            complaint=complaint,
            translated_complaint=None
        )

    try:

        result = process_all_complaints(complaint)

        translated_complaint = " | ".join(
            result["translation_results"]
        )

        if result["recommendations"]:

            first_recommendation = result["recommendations"][0]

            result["specialty"] = first_recommendation["specialty"]

            result["poliklinik"] = first_recommendation.get(
                "poliklinik"
            )

        else:

            raise ValueError(
                "Tidak ditemukan rekomendasi spesialisasi."
            )

        confidence = result["recommendations"][0].get(
            "confidence"
        )

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            error=None,
            complaint=complaint,
            translated_complaint=translated_complaint
        )

    except Exception as error:

        app.logger.exception(
            "Terjadi kesalahan saat memproses keluhan."
        )

        return render_template(
            "result.html",
            result=None,
            confidence=None,
            error=(
                "Keluhan belum dapat diproses. "
                f"Detail error: {str(error)}"
            ),
            complaint=complaint,
            translated_complaint=None
        )


# =========================================================
# 16. DOCTOR DIRECTORY
# =========================================================

@app.route("/dokter", methods=["GET"])
def dokter():

    poliklinik = request.args.get(
        "poliklinik",
        ""
    ).strip()

    if not poliklinik:
        return "Nama poliklinik tidak ditemukan.", 400

    doctor_list = get_doctors_by_polyclinic(
        poliklinik
    )

    return render_template(
        "dokter.html",
        poliklinik=poliklinik,
        doctors=doctor_list,
        hospital_name="RS Stella Maris Makassar",
        phone="0813 60 888 100 / 0411-871391",
        whatsapp="0813 98 888 200",
        registration_url=(
            "https://rsstellamaris.com/"
            "registrasi-online-rs-stella-maris/"
        )
    )


# =========================================================
# 17. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
