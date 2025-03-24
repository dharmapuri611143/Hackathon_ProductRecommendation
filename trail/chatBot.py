import streamlit as st
from openai import OpenAI
from transformers import pipeline
import PyPDF2
import docx
import os
import configparser
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import tempfile
import whisper
import soundfile as sf

# --- Configuration ---
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config['openai']['api_key']
client = OpenAI(api_key=openai_api_key)

# --- Page Setup ---
st.set_page_config(page_title="🧠 Smart Chatbot", layout="wide")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #eef2f3, #8e9eab);
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background: linear-gradient(to right, #eef2f3, #8e9eab);
        padding: 20px;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stTextArea textarea {
        background-color: #f7f9fb;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4169E1;
        color: white;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 16px;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2c4cb2;
        transform: scale(1.03);
    }
    .stRadio > div {
        background-color: #f0f4f7;
        padding: 10px;
        border-radius: 10px;
    }
    .stFileUploader {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🧠 Smart Recommendation Chatbot - Text, Live Audio, Files")

# --- Hugging Face Model ---
@st.cache_resource
def load_hf_pipeline():
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", device=0)

# --- File Reader ---
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8")

# --- Audio Processor ---
class MyAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame.to_ndarray())
        return frame

# --- Save Audio to File ---
def save_audio_to_wav(audio_frames, sample_rate=48000):
    audio_data = np.concatenate(audio_frames, axis=1).flatten().astype(np.float32)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio_data, sample_rate)
    return temp_wav.name

# --- UI Inputs ---
text_prompt = st.text_area("💬 Enter your text prompt:", height=150)

uploaded_file = st.file_uploader("📎 Upload a file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

st.markdown("🎙️ **Record Audio Prompt:**")
ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=MyAudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

use_model = st.radio("🧠 Choose a model:", ["OpenAI GPT-3.5", "Hugging Face Mistral"])

if st.button("💬 Generate Response"):
    context = ""

    # Add file content
    if uploaded_file:
        context += extract_text_from_file(uploaded_file) + "\n\n"

    # Transcribe Audio
    transcribed_text = ""
    if ctx and ctx.audio_processor and ctx.audio_processor.audio_frames:
        st.success("🎤 Audio recorded! Transcribing...")

        wav_file = save_audio_to_wav(ctx.audio_processor.audio_frames)
        model = whisper.load_model("base")
        result = model.transcribe(wav_file)
        transcribed_text = result["text"]

        st.markdown("📝 **Transcribed Audio:**")
        st.write(transcribed_text)

        context += transcribed_text + "\n\n"
    else:
        st.info("No audio input detected.")

    # Final prompt
    full_prompt = context + "\nUser Prompt:\n" + text_prompt

    st.subheader("🤖 Response")
    if use_model == "OpenAI GPT-3.5":
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI Error: {e}")
    else:
        try:
            pipe = load_hf_pipeline()
            output = pipe(full_prompt, max_new_tokens=500)[0]["generated_text"]
            st.write(output.split("PROMPT ENDED:")[-1] if "PROMPT ENDED:" in output else output)
        except Exception as e:
            st.error(f"Hugging Face Error: {e}")
