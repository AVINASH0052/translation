# Language Translation App

This project is a **Language Translation Application** built with Python, Streamlit, and the Hugging Face Transformers library. It supports text and image input for detecting languages and translating text into multiple target languages.

## Features
- Detect the language of the input text using `langdetect`.
- Translate text between supported languages using pre-trained models from the Helsinki-NLP project.
- Extract text from images using Tesseract OCR (`pytesseract`).
- Supports translation for multiple languages including English, Hindi, French, German, Spanish, Chinese, Japanese, Russian, and Italian.

## Requirements
Ensure that the following dependencies are installed:
- `streamlit`: For creating the web application interface.
- `transformers`: For utilizing pre-trained language models.
- `langdetect`: For detecting the language of the input text.
- `pytesseract`: For extracting text from images.
- `pillow`: For handling image uploads.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/language-translation-app.git
   cd language-translation-app
