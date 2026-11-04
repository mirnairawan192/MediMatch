
import os
import re
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import nltk
import torch

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.models import load_model

from langdetect import detect, LangDetectException, DetectorFactory


# =========================================================
# 1. CONFIGURATION
# =========================================================

DetectorFactory.seed = 0

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# 2. FILE PATHS
# =========================================================

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
            f"File tidak ditemukan: {os.path.basename(file_path)}\n"
            f"Lokasi: {file_path}"
        )


# =========================================================
# 4. LOAD DOC2VEC MODEL
# =========================================================

print("\nMemuat model Doc2Vec...")

doc2vec_model = Doc2Vec.load(DOC2VEC_PATH)

print("Doc2Vec berhasil dimuat.")


# =========================================================
# 5. LOAD LSTM MODEL
# =========================================================

print("\nMemuat model LSTM...")

lstm_model = load_model(LSTM_PATH)

print("LSTM berhasil dimuat.")


# =========================================================
# 6. LOAD LABEL ENCODER
# =========================================================

print("\nMemuat Label Encoder...")

with open(ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

print("Label Encoder berhasil dimuat.")


# =========================================================
# 7. VALIDATE MODEL DIMENSIONS
# =========================================================

if doc2vec_model.vector_size != 150:

    raise ValueError(
        "Ukuran vektor Doc2Vec harus 150."
    )


if lstm_model.input_shape[-1] != 150:

    raise ValueError(
        "Ukuran input LSTM tidak sesuai dengan Doc2Vec."
    )


if lstm_model.output_shape[-1] != len(label_encoder.classes_):

    raise ValueError(
        "Jumlah output LSTM tidak sesuai dengan Label Encoder."
    )


print("\nSeluruh model berhasil dimuat.")
print("Jumlah kelas:", len(label_encoder.classes_))
print("Daftar kelas:", list(label_encoder.classes_))


# =========================================================
# 8. LOAD SPECIALTY MAPPING
# =========================================================

print("\nMemuat mapping spesialisasi...")

mapping_df = pd.read_csv(
    MAPPING_PATH,
    encoding="utf-8-sig"
)

mapping_df.columns = (
    mapping_df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

required_columns = {
    "kaggle_specialty",
    "rs_poliklinik"
}

if not required_columns.issubset(mapping_df.columns):

    raise ValueError(
        "Kolom mapping tidak sesuai.\n"
        "Kolom wajib: kaggle_specialty dan rs_poliklinik."
    )


mapping_df = mapping_df.fillna("")

specialty_mapping = {

    str(row["kaggle_specialty"]).strip().lower():
    str(row["rs_poliklinik"]).strip()

    for _, row in mapping_df.iterrows()

    if str(row["kaggle_specialty"]).strip()

}

print("Mapping berhasil dimuat.")
print("Jumlah mapping:", len(specialty_mapping))


# =========================================================
# 9. LOAD DOCTOR DATABASE
# =========================================================

print("\nMemuat database dokter...")

doctors_df = pd.read_csv(
    DOCTORS_PATH,
    encoding="utf-8-sig"
)

doctors_df.columns = (
    doctors_df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

required_doctor_columns = {
    "poliklinik",
    "nama_dokter"
}

if not required_doctor_columns.issubset(doctors_df.columns):

    raise ValueError(
        "Kolom database dokter tidak sesuai.\n"
        "Kolom wajib: poliklinik dan nama_dokter."
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

print("Database dokter berhasil dimuat.")
print("Jumlah data dokter:", len(doctors_df))


# =========================================================
# 10. NORMALIZE POLYCLINIC NAME
# =========================================================

def normalize_polyclinic(name):

    name = str(name).strip().upper()

    name = re.sub(r"\s+", " ", name)

    return name


# =========================================================
# 11. GET DOCTORS BY POLYCLINIC
# =========================================================

def get_doctors_by_polyclinic(polyclinic):

    target_polyclinic = normalize_polyclinic(
        polyclinic
    )

    filtered_doctors = doctors_df[
        doctors_df["poliklinik"].apply(
            normalize_polyclinic
        ) == target_polyclinic
    ]

    doctor_list = filtered_doctors[
        "nama_dokter"
    ].drop_duplicates().tolist()

    return doctor_list


# =========================================================
# 12. LOAD NLTK RESOURCES
# =========================================================

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# =========================================================
# 13. LOAD TRANSLATION MODEL
# =========================================================

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-id-en"

print("\nMemuat model penerjemah...")

translator_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL
)

translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)

translator_model.eval()

print("Model penerjemah berhasil dimuat.")


# =========================================================
# 14. TRANSLATE INDONESIAN TO ENGLISH
# =========================================================

@lru_cache(maxsize=256)
def translate_to_english(text):

    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = translator_model.generate(
            **inputs,
            max_new_tokens=128
        )

    translated_text = translator_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if not translated_text.strip():

        raise ValueError("Hasil terjemahan kosong.")

    return translated_text.strip()


# =========================================================
# 15. LANGUAGE DETECTION
# =========================================================

def detect_complaint_language(text):

    try:

        return detect(text)

    except LangDetectException:

        return "unknown"


# =========================================================
# 16. TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # Text cleaning
    text = re.sub(
        r"[^\w\s]",
        " ",
        str(text)
    )

    # Case folding
    text = text.lower()

    # Tokenizing
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


# =========================================================
# 17. SPLIT MULTIPLE COMPLAINTS
# =========================================================

def split_complaints(complaint):

    complaint = complaint.strip()

    parts = re.split(
        r"\s+(?:and|but|also|while|whereas|as well as)\s+",
        complaint,
        flags=re.IGNORECASE
    )

    valid_parts = []

    for part in parts:

        part = part.strip()

        tokens = preprocess_text(part)

        if len(tokens) >= 1:

            valid_parts.append(part)

    if not valid_parts:

        return [complaint]

    print("\n======================================")
    print("PEMISAHAN KELUHAN")
    print("======================================")

    for i, part in enumerate(valid_parts, start=1):

        print(f"Keluhan {i}: {part}")

    print("======================================\n")

    return valid_parts


# =========================================================
# 18. PREDICT ONE COMPLAINT
# =========================================================

def predict_one_complaint(complaint):

    tokens = preprocess_text(complaint)

    if not tokens:

        raise ValueError(
            "Keluhan tidak memiliki kata yang dapat diproses."
        )

    # DOC2VEC INFERENCE

    vector = doc2vec_model.infer_vector(
        tokens,
        epochs=40
    )

    vector = np.asarray(
        vector,
        dtype=np.float32
    ).reshape(1, 1, 150)

    # LSTM PREDICTION

    probabilities = lstm_model.predict(
        vector,
        verbose=0
    )[0]

    if len(probabilities) != len(label_encoder.classes_):

        raise ValueError(
            "Output LSTM tidak sesuai dengan jumlah kelas."
        )

    # GET HIGHEST PREDICTION

    predicted_index = int(
        np.argmax(probabilities)
    )

    specialty = str(
        label_encoder.inverse_transform(
            [predicted_index]
        )[0]
    ).strip().lower()

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return {
        "specialty": specialty,
        "confidence": confidence,
        "complaint": complaint
    }


# =========================================================
# 19. PREDICT MULTIPLE SPECIALTIES
# =========================================================

def predict_specialties(complaint):

    complaint_parts = split_complaints(complaint)

    recommendations = []

    for part in complaint_parts:

        prediction = predict_one_complaint(part)

        print(
            f"Keluhan: {part} | "
            f"Prediksi: {prediction['specialty']} | "
            f"Confidence: {prediction['confidence']:.2f}%"
        )

        # CHECK DUPLICATE SPECIALTY

        existing = next(
            (
                item for item in recommendations
                if item["specialty"] == prediction["specialty"]
            ),
            None
        )

        if existing is None:

            recommendations.append(prediction)

    # HANDLE EMERGENCY

    emergency_prediction = [
        item for item in recommendations
        if item["specialty"] == "emergency"
    ]

    if emergency_prediction:

        return {
            "type": "emergency",
            "recommendations": [],
            "all_predictions": recommendations
        }

    print("\n======================================")
    print("HASIL AKHIR REKOMENDASI")
    print("======================================")

    for i, item in enumerate(recommendations, start=1):

        print(
            f"Rekomendasi {i}: {item['specialty']} "
            f"({item['confidence']:.2f}%)"
        )

    print("======================================\n")

    return {
        "type": "normal",
        "recommendations": recommendations
    }


# =========================================================
# 20. DETECT EYE AND DENTAL COMPLAINTS
# =========================================================

def detect_specialty_override(complaint):

    complaint_lower = str(complaint).lower()

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

    # DENTAL COMPLAINTS

    if any(
        re.search(pattern, complaint_lower)
        for pattern in dental_keywords
    ):

        return {
            "specialty": "dentistry",
            "poliklinik": "GIGI"
        }

    # EYE COMPLAINTS

    if any(
        re.search(pattern, complaint_lower)
        for pattern in eye_keywords
    ):

        return {
            "specialty": "ophthalmology",
            "poliklinik": "MATA"
        }

    return None


# =========================================================
# 21. BUILD RULE-BASED RECOMMENDATION
# =========================================================

def build_rule_based_result(recommendation):

    mapped_recommendation = {

        "specialty": recommendation["specialty"],

        "confidence": None,

        "poliklinik": recommendation["poliklinik"],

        "status": "normal"

    }

    return {

        "type": "normal",

        "recommendations": [mapped_recommendation],

        "specialty": mapped_recommendation["specialty"],

        "poliklinik": mapped_recommendation["poliklinik"]

    }


# =========================================================
# 22. MAP SPECIALTIES TO HOSPITAL POLYCLINICS
# =========================================================

def map_recommendations(recommendations):

    mapped_recommendations = []

    for recommendation in recommendations:

        specialty = recommendation["specialty"]

        confidence = recommendation["confidence"]

        poliklinik = specialty_mapping.get(
            specialty,
            ""
        )

        if (
            not poliklinik
            or poliklinik.strip().upper() == "PENDING"
        ):

            status = "unmapped"

            poliklinik = None

        else:

            status = "normal"

        mapped_recommendations.append({

            "specialty": specialty,

            "confidence": confidence,

            "poliklinik": poliklinik,

            "status": status

        })

    return mapped_recommendations


# =========================================================
# 23. MAIN PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    # GET: TAMPILKAN HALAMAN UTAMA

    if request.method == "GET":

        return render_template("index.html")


    # POST: PROSES KELUHAN

    complaint = request.form.get(
        "complaint",
        ""
    ).strip()

    result = None

    confidence = None

    error = None

    translated_complaint = None


    if not complaint:

        error = "Silakan masukkan keluhan terlebih dahulu."

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            error=error,
            complaint=complaint,
            translated_complaint=translated_complaint
        )


    try:

        # ----------------------------------
        # CHECK RULE-BASED EYE / DENTAL ROUTING
        # ----------------------------------

        special_recommendation = detect_specialty_override(
            complaint
        )

        if special_recommendation:

            print(
                "\nRule-based recommendation:",
                special_recommendation["specialty"],
                "->",
                special_recommendation["poliklinik"]
            )

            result = build_rule_based_result(
                special_recommendation
            )


        else:

            # ----------------------------------
            # DETECT LANGUAGE
            # ----------------------------------

            detected_language = detect_complaint_language(
                complaint
            )

            print("\nInput complaint:", complaint)

            print(
                "Detected language:",
                detected_language
            )


            # ----------------------------------
            # TRANSLATE INPUT
            # ----------------------------------

            if detected_language == "en":

                translated_complaint = complaint

            else:

                translated_complaint = translate_to_english(
                    complaint
                )

            print(
                "\nTranslated complaint:",
                translated_complaint
            )


            # ----------------------------------
            # PREDICT SPECIALTIES
            # ----------------------------------

            prediction_output = predict_specialties(
                translated_complaint
            )


            # ----------------------------------
            # HANDLE EMERGENCY
            # ----------------------------------

            if prediction_output["type"] == "emergency":

                result = {

                    "type": "emergency",

                    "recommendations": [],

                    "specialty": "emergency",

                    "poliklinik": None

                }


            else:

                # ----------------------------------
                # MAP RECOMMENDATIONS
                # ----------------------------------

                mapped_recommendations = map_recommendations(
                    prediction_output["recommendations"]
                )

                if not mapped_recommendations:

                    raise ValueError(
                        "Tidak ditemukan rekomendasi spesialisasi."
                    )


                first_recommendation = mapped_recommendations[0]


                # ----------------------------------
                # DETERMINE RESULT TYPE
                # ----------------------------------

                if any(
                    recommendation["status"] == "unmapped"
                    for recommendation in mapped_recommendations
                ):

                    result_type = "unmapped"

                else:

                    result_type = "normal"


                # ----------------------------------
                # SAVE RESULT
                # ----------------------------------

                result = {

                    "type": result_type,

                    "recommendations": mapped_recommendations,

                    "specialty": first_recommendation["specialty"],

                    "poliklinik": first_recommendation["poliklinik"]

                }

                confidence = first_recommendation["confidence"]


    except Exception as e:

        app.logger.exception(
            "Terjadi kesalahan saat memproses keluhan."
        )

        error = (
            "Keluhan belum dapat diproses. "
            f"Detail error: {str(e)}"
        )


    # ----------------------------------
    # RENDER RESULT PAGE
    # ----------------------------------

    return render_template(

        "result.html",

        result=result,

        confidence=confidence,

        error=error,

        complaint=complaint,

        translated_complaint=translated_complaint

    )


# =========================================================
# 24. DOCTOR DIRECTORY PAGE
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
# 25. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )
