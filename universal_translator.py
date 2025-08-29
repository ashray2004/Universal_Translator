import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import tempfile
import os
import ffmpeg

# ----------------- SIGN LANGUAGE CLASSIFIER SETUP ----------------- #
@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(image: Image.Image, mode: str, size):
    image = image.resize(size)
    if mode == "grayscale":
        image = image.convert("L")
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=-1)
    else:
        image = image.convert("RGB")
        image_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image_array, axis=0)

model_map = {
    "Digit Sign Language": {
        "path": "models//digitSignLanguage.h5",
        "labels": [str(i) for i in range(10)],
        "mode": "rgb"
    },
    "ASL Characters": {
        "path": "models/americanSignLanguage.h5",
        "labels": [chr(i) for i in range(65, 91)],
        "mode": "grayscale"
    },
    "ISL Characters": {
        "path": "models/indianSignLanguage.h5",
        "labels": [chr(i) for i in range(65, 91)],
        "mode": "rgb"
    }
}

# ----------------- TRANSLATOR SETUP ----------------- #
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_nllb_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

WHISPER_LANG_TO_NLLB = {
    "en": "eng_Latn", "hi": "hin_Deva", "fr": "fra_Latn", "es": "spa_Latn",
    "de": "deu_Latn", "bn": "ben_Beng", "gu": "guj_Gujr", "ta": "tam_Taml",
    "te": "tel_Telu", "ar": "arb_Arab", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "ru": "rus_Cyrl", "ur": "urd_Arab",
}

LANGS = {
    "English": "eng_Latn", "Hindi": "hin_Deva", "French": "fra_Latn", "Spanish": "spa_Latn",
    "German": "deu_Latn", "Arabic": "arb_Arab", "Bengali": "ben_Beng", "Gujarati": "guj_Gujr",
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan", "Russian": "rus_Cyrl", "Urdu": "urd_Arab"
}

def transcribe_with_whisper(audio_path):
    model = load_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"], result["language"]

def extract_audio_from_video(video_file_path):
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    ffmpeg.input(video_file_path).output(temp_audio.name).run(quiet=True, overwrite_output=True)
    return temp_audio.name

def translate_text(text, source_lang_code, target_lang_code):
    tokenizer, model = load_nllb_model()
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
    )
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# ----------------- STREAMLIT UI ----------------- #
st.set_page_config(page_title="Universal Translator", layout="centered")
st.title("🧠 Universal Translator App")

tabs = st.tabs(["📷 Sign Language Classifier", "🌍 Language Translator"])

# ========== SIGN LANGUAGE TAB ==========
with tabs[0]:
    st.subheader("📷 Sign Language Image Classifier")
    model_choice = st.selectbox("Choose the model:", list(model_map.keys()))
    model_info = model_map[model_choice]
    model = load_tf_model(model_info["path"])
    input_shape = model.input_shape[1:3]

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="sign")
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_container_width=False)
        processed_image = preprocess_image(image, model_info["mode"], input_shape)
        prediction = model.predict(processed_image)
        predicted_label = model_info["labels"][np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.markdown(f"### ✅ Prediction: **{predicted_label}**")
        st.markdown(f"📊 Confidence: **{confidence:.2f}%**")

# ========== LANGUAGE TRANSLATOR TAB ==========
with tabs[1]:
    st.subheader("🌍 Language-to-Language Translator")
    input_type = st.radio("Select input type", ("Text", "Audio", "Video"))
    target_lang = st.selectbox("Select target language", list(LANGS.keys()))
    target_lang_code = LANGS[target_lang]

    uploaded_file = None
    input_text = ""
    transcribed_text = ""
    source_lang_code = "eng_Latn"

    if input_type == "Text":
        input_text = st.text_area("Enter text to translate:")
    elif input_type == "Audio":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"], key="audio")
    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4"], key="video")

    if st.button("Translate", key="translate"):
        with st.spinner("Processing..."):
            if input_type == "Text":
                transcribed_text = input_text
                source_lang_code = "eng_Latn"
            elif uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                if input_type == "Video":
                    audio_path = extract_audio_from_video(tmp_path)
                else:
                    audio_path = tmp_path

                transcribed_text, whisper_lang = transcribe_with_whisper(audio_path)
                source_lang_code = WHISPER_LANG_TO_NLLB.get(whisper_lang, "eng_Latn")
                os.remove(audio_path)

        st.markdown("#### 📝 Transcription / Input:")
        st.info(transcribed_text)

        translation = translate_text(transcribed_text, source_lang_code, target_lang_code)
        st.markdown(f"#### 🌐 Translated to {target_lang}:")
        st.success(translation)
