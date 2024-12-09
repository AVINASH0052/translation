import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import pytesseract
from PIL import Image
import re

# Load the translation model
@st.cache_resource
def load_translation_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Supported languages for translation
language_mapping = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "it": "Italian",
}

# Available language pairs for translation (Helsinki-NLP models)
supported_pairs = {
    "en-hi", "hi-en", "en-fr", "fr-en", "en-de", "de-en", "en-es", "es-en",
    "en-zh", "zh-en", "en-ja", "ja-en", "en-ru", "ru-en", "en-it", "it-en",
}

# Function to split text into paragraphs
def split_text_into_paragraphs(text):
    """Splits the input text into paragraphs."""
    paragraphs = text.split("\n")
    return paragraphs

# Function to split a paragraph into sentences
def split_paragraph_into_sentences(paragraph):
    """Splits a paragraph into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    return sentences

# Function to translate sentences within a paragraph
def translate_paragraph(paragraph, model, tokenizer):
    """Translates a paragraph sentence by sentence and recombines."""
    sentences = split_paragraph_into_sentences(paragraph)
    translations = []
    for sentence in sentences:
        try:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_length=512)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation)
        except Exception as e:
            translations.append(f"[Error in translation: {e}]")
    return " ".join(translations)

# Extract text from an uploaded image using pytesseract
def extract_text_from_image(image):
    """Extracts text from an uploaded image using Tesseract OCR."""
    try:
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"

# App title
st.title("Language Translation App")

# File uploader for image upload
uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

# Text extraction from image if uploaded
if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        extracted_text = extract_text_from_image(image)
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    extracted_text = ""

# Text input
text = st.text_area("Enter text to translate", extracted_text)

# Language detection
if text:
    detected_lang = detect(text)
    detected_lang_name = language_mapping.get(detected_lang, "Unknown")
    st.write(f"Detected Language: {detected_lang_name} ({detected_lang})")

# Language selection
target_lang = st.selectbox("Select Target Language", options=language_mapping.keys(), format_func=lambda x: language_mapping[x])

# Translate button
if st.button("Translate"):
    if detected_lang == target_lang:
        st.write("The source and target languages are the same. No translation needed.")
    else:
        translation_pair = f"{detected_lang}-{target_lang}"
        if translation_pair not in supported_pairs:
            st.error(f"Translation from {language_mapping.get(detected_lang, 'Unknown')} to {language_mapping.get(target_lang, 'Unknown')} is not supported.")
        else:
            try:
                # Load specific translation model
                translation_model = f"Helsinki-NLP/opus-mt-{detected_lang}-{target_lang}"
                tokenizer, model = load_translation_model(translation_model)

                # Split text into paragraphs
                paragraphs = split_text_into_paragraphs(text)

                # Translate each paragraph
                translated_paragraphs = []
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Ignore empty paragraphs
                        translated_paragraph = translate_paragraph(paragraph, model, tokenizer)
                        translated_paragraphs.append(translated_paragraph)

                # Combine paragraphs with line breaks
                full_translation = "\n\n".join(translated_paragraphs)

                # Display full translation
                st.write(f"Translation in {language_mapping[target_lang]}:")
                st.write(full_translation)

            except Exception as e:
                st.error(f"An error occurred: {e}")
