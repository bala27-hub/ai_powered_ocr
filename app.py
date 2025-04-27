import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from g4f.client import Client
from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator
import re

# Page Configuration (MUST BE FIRST)
st.set_page_config(page_title="Tamil OCR GPT App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stDownloadButton>button {background-color: #008CBA; color: white; border-radius: 8px;}
    .stTextInput>input, .stTextArea>textarea {border-radius: 8px; border: 1px solid #ddd;}
    .stSpinner {color: #4CAF50;}
    h1 {color: #2c3e50; font-family: 'Arial', sans-serif;}
    h2, h3 {color: #34495e;}
    .sidebar .sidebar-content {background-color: #ecf0f1;}
    .feature-box {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("<style>.main {background-color: #2c3e50; color: #ecf0f1;} h1, h2, h3 {color: #ecf0f1;}</style>", unsafe_allow_html=True)
    st.write("Adjust settings for a personalized experience.")
    st.slider("Font Size", 12, 24, 16, key="font_size")
    st.color_picker("Highlight Color", "#4CAF50", key="highlight_color")

# Main Title and Description
st.title("🖋️ Tamil OCR + Ai Model + TTS + Q&A")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        Upload a Tamil text image to extract, enhance, translate, and interact with the content in style!
    </div>
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Tamil Image", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

if uploaded_file:
    # Progress Bar
    progress_bar = st.progress(0)

    # Image Processing
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="📷 Original Image", use_column_width=True)
    progress_bar.progress(20)

    # Convert to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Check and upscale resolution if too small
    height, width = cv_image.shape[:2]
    if height < 300 or width < 300:
        scale_factor = max(300 / height, 300 / width)
        cv_image = cv2.resize(cv_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        st.write(f"Image upscaled by {scale_factor:.2f}x for better OCR accuracy.")

    # Minimal Preprocessing
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display preprocessed image
    preprocessed_image = Image.fromarray(binary)
    with col2:
        st.image(preprocessed_image, caption="🛠️ Preprocessed Image", use_column_width=True)
    progress_bar.progress(40)

    # OCR Extraction with Optimized Configuration
    with st.expander("📄 Raw OCR Output", expanded=True):
        custom_config = r'--oem 3 --psm 3 -l tam'  # OEM 3 (LSTM), PSM 3 (auto page segmentation)
        extracted_text = pytesseract.image_to_string(binary, lang='tam', config=custom_config)
        data = pytesseract.image_to_data(binary, lang='tam', config=custom_config, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        average_conf = round(np.mean(confidences), 2) if confidences else 0  # Fixed line
        st.markdown(f"**🧠 Tesseract Confidence:** `{average_conf}%`")
        st.text_area("Raw Extracted Text", value=extracted_text, height=200)
    progress_bar.progress(60)

    # GPT Enhancement
    with st.spinner("🔮 Enhancing with GPT-4o-mini..."):
        client = Client()
        prompt = f"""
Below is Tamil text extracted from an image using OCR:

{extracted_text}

Please:
1. Correct OCR errors.
2. Format the text with headings (###), bullet points, and structure.
3. Keep it in Tamil.
4. At the end, give a confidence score from 1–10.

Format:
---
Enhanced Text:
### Title
- Bullet
  - Subpoint
...

Confidence: <number>
"""
        try:
            response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], web_search=False)
            full_response = response.choices[0].message.content
            enhanced_match = re.search(r"Enhanced Text:(.*?)(?:Confidence:|$)", full_response, re.DOTALL)
            confidence_match = re.search(r"Confidence:\s*(\d+)", full_response)
            enhanced_text = enhanced_match.group(1).strip() if enhanced_match else full_response
            gpt_confidence = confidence_match.group(1) if confidence_match else "N/A"
        except Exception as e:
            enhanced_text = f"❌ GPT Enhancement Failed: {e}"
            gpt_confidence = "N/A"
    progress_bar.progress(80)

    # Enhanced Text Display
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.subheader("✨ Enhanced Tamil Text")
    st.markdown(f"**🤖 GPT Confidence:** <span style='color:{st.session_state.highlight_color}'>{gpt_confidence}/10</span>", unsafe_allow_html=True)
    sentences = re.split(r'(?<=[.!?]) +', enhanced_text)
    chunks = [' '.join(sentences[i:i + 4]) for i in range(0, len(sentences), 4)]
    for chunk in chunks:
        st.markdown(chunk)
    st.markdown("</div>", unsafe_allow_html=True)

    # Translation and TTS in Columns
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.subheader("🌐 English Translation")
        try:
            translated_text = GoogleTranslator(source='ta', target='en').translate(enhanced_text)
            st.markdown(f"<div style='background-color:#eef6ff; padding:10px; border-radius:10px'>{translated_text}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Translation Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
        st.subheader("🔊 Tamil Text-to-Speech")
        try:
            tts = gTTS(enhanced_text, lang='ta')
            tts_audio = BytesIO()
            tts.write_to_fp(tts_audio)
            tts_audio.seek(0)
            st.audio(tts_audio, format="audio/mp3")
        except Exception as e:
            st.warning(f"TTS Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Editable Text and Download
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.subheader("✍️ Edit or Copy Enhanced Text")
    edited_text = st.text_area("", value=enhanced_text, height=200)
    st.download_button("📥 Download Enhanced Text", data=edited_text, file_name="enhanced_tamil_text.txt", mime="text/plain")
    st.markdown("</div>", unsafe_allow_html=True)
    progress_bar.progress(100)

    # Chatbot Section
    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
    st.subheader("💬 Chat About the Text")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant who answers based on the enhanced Tamil text."},
            {"role": "user", "content": f"The extracted Tamil content is:\n{enhanced_text}"}
        ]
    question = st.text_input("Ask a question (Tamil or English)", key="chat_input")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("💡 Thinking..."):
            try:
                chat_response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.chat_history, web_search=False)
                answer = chat_response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.markdown(f"**🧠 Answer:** {answer}")
            except Exception as e:
                st.error(f"Chatbot Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding-top: 20px;'>
        Made with ❤️ by Tamil OCR Team | Powered by AI
    </div>
""", unsafe_allow_html=True)