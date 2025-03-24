import streamlit as st
import pandas as pd
import configparser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import speech_recognition as sr
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer, util
# Read OpenAI API key from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config['openai']['api_key']

# Initialize ChatOpenAI Model
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Load datasets
customer_profiles = pd.read_csv("customer_profiles.csv")
transactions = pd.read_csv("transaction_history.csv")
social_media = pd.read_csv("social_media_activity.csv")
demographics = pd.read_csv("demographic_details.csv")

# Enhanced Prompt Template with product names and formatting
template = """
Given the following detailed customer profile:

Demographic Information:
{demographics}

Recently Purchased Products:
{transactions}

Social Media Interests and Activities:
{social_media}

Please provide the following clearly referencing the provided customer data:

1. **Adaptive Recommendation**  
- Suggest 1 specific product or service that adapts to a recent shift in the customer's behavior.  
- Include a real-world brand or product name (e.g., "Netflix Premium", "Samsung SmartThings Starter Kit").  
- Explain the connection to the customer's latest transactions.

2. **Generated Personalized Suggestions**  
- Recommend at least 2 highly relevant products or services.  
- Include specific examples with names (e.g., "Tata AIA Term Plan", "Amazon Echo Show", etc).  
- Clearly explain how the suggestion connects with demographics or social behavior.

3. **Sentiment-Driven Content Recommendation**  
- Based on social media sentiment, recommend one piece of educational or promotional content (e.g., "YouTube video: 5 Ways to Save in 2024", "Blog: How to Budget with Kids", etc).  
- Explain how it helps the customer based on their social posts.

Format the output using headings and bullet points.
"""

prompt = PromptTemplate.from_template(template)

# LangChain Runnable Sequence
recommendation_chain = (
        {"demographics": RunnablePassthrough(),
         "transactions": RunnablePassthrough(),
         "social_media": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="✨ AI Personalized Recommendations ✨", layout="wide")
st.title("✨ AI-Powered Personalized Recommendation Engine")


tabs = st.tabs(["🔎 Existing Customer", "🆕 New Customer", "🤖 Ask Bot"])

# Existing Customer Tab
with tabs[0]:
    st.header("🔍 Existing Customer Profile")
    customer_id = st.selectbox("Select Customer ID:", customer_profiles['Customer_Id'].unique())

    # Fetch data
    demo = demographics[demographics['Customer_Id'] == customer_id].iloc[0]
    customer_trans = transactions[transactions['Customer_Id'] == customer_id][['Purchase_Date', 'Category', 'Amount (In Dollars)']]
    customer_social = social_media[social_media['Customer_Id'] == customer_id][['Timestamp', 'Content']]

    # Display data in tables
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Demographics")
        st.table(demo.drop("Customer_Id"))

    with col2:
        st.subheader("🛒 Recent Transactions")
        st.dataframe(customer_trans.reset_index(drop=True), height=200)

    with col3:
        st.subheader("📲 Social Media Activities")
        st.dataframe(customer_social.reset_index(drop=True), height=200)

    if st.button("🚀 Generate Recommendations", key="existing"):
        with st.spinner("Analyzing data and generating personalized recommendations..."):
            recommendations = recommendation_chain.invoke({
                "demographics": demo.drop("Customer_Id").to_dict(),
                "transactions": ", ".join(customer_trans['Category'].tolist()),
                "social_media": ", ".join(customer_social['Content'].tolist())
            })
        st.success("✅ Recommendations Generated!")
        st.subheader("🎯 Personalized Recommendations")
        st.markdown(recommendations, unsafe_allow_html=True)

# New Customer Tab
with tabs[1]:
    st.header("✨ Enter New Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Demographic Information")
        name = st.text_input("Enter your name:")
        age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
        country = st.selectbox("Location", ["India","USA", "Canada", "UK", "Australia"])
        preference = st.selectbox("Preference", ["Online Shopping","Hybrid", "In-Store"])
        interests = st.selectbox("Interests", ["Online Shopping","Hybrid", "In-Store"])
        json_data = {
            "Customer Name": name,
            "Country": country,
            "Age": age,
            "Preference":preference,
            "Interests":interests
        }
        #st.subheader("Generated JSON Data")
        #st.json(json_data)
       # demo_input = st.text_area("Demographics (JSON format)", height=200, value=json.dumps(json_data, indent=4))

    with col2:
        st.subheader("🛒 Recent Transactions & 📲 Social Media")
        # Additional selection boxes
        category_filter = st.selectbox("Filter by Transaction Category:", ["All"] + transactions['Category'].unique().tolist())
        social_filter = st.selectbox("Filter by Social Media Platform:", ["All", "Facebook", "Twitter", "Instagram"])

        customer_trans = transactions[((transactions['Category'] == category_filter) | (category_filter == "All"))][['Purchase_Date', 'Category', 'Amount (In Dollars)']]
        customer_social = social_media[(social_media['Customer_Id'] == customer_id) &
                                   ((social_media['Platform'] == social_filter) | (social_filter == "All"))][['Timestamp', 'Content']]


    if st.button("🚀 Generate Recommendations", key="new"):
        try:
            demo_dict = json_data
        except json.JSONDecodeError:
            st.error("Demographic info JSON is invalid. Please correct it.")
            st.stop()

        with st.spinner("Generating personalized recommendations..."):
            recommendations_new = recommendation_chain.invoke({
                "demographics": demo_dict,
                "transactions":  ", ".join(customer_trans['Category'].tolist()),
                "social_media":  ", ".join(customer_social['Content'].tolist())
            })
        st.success("✅ Recommendations Generated for New Customer!")
        st.subheader("🎯 Personalized Recommendations")
        st.markdown(recommendations_new, unsafe_allow_html=True)

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess and combine all text data
def combine_text_data(transactions, social_media, customer_profiles, demographics):
    combined_data = pd.concat([
        transactions['Category'],
        social_media['Content'],
        customer_profiles.astype(str).apply(' '.join, axis=1),
        demographics.astype(str).apply(' '.join, axis=1)
    ])
    return combined_data

# Function to find top matches

# Function to find the best match
def get_best_match(user_input, all_texts, model):
    # Compute embeddings
    all_text_embeddings = model.encode(all_texts.tolist(), convert_to_tensor=True)
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, all_text_embeddings)[0]
    best_index = similarities.argmax().item()  # Get the index of the highest similarity
    best_match = all_texts.iloc[best_index]
    return best_match


with tabs[2]:
    st.header("🤖 Ask Bot")
    st.subheader("Chat with AI - Text, Audio, and Image Input")

    user_input = ""

    # Text input
    text_input = st.text_area("Type your query here:", height=100)

    # Audio recording
    if st.button("🎙️"):
        st.info("Recording... Please speak.")


        fs = 44100  # Sample rate
        duration = 5  # Duration of recording in seconds

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished

        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        write(temp_audio_path, fs, recording)

        st.success("Recording complete!")


        recognizer = sr.Recognizer()

        # Record audio from the microphone
        with sr.Microphone() as source:
            print("Speak now...")
            audio_data = recognizer.listen(source)
            print("Processing...")

            try:
                # Convert speech to text
                text = recognizer.recognize_google(audio_data)
                print("Text Output:", text)
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError:
                print("Could not request results; check your internet connection.")

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    # Title for the web app
    st.title("Image to Text Converter")

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if text_input:
        user_input = text_input
    elif uploaded_file:
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Extract text from the image using PaddleOCR
        results = ocr.ocr(temp_image_path, cls=True)
        extracted_text = " ".join([line[1][0] for line in results[0]])


        # Use the extracted text as user input
        user_input = extracted_text.strip()
        st.subheader("User Input:")
        st.write(user_input)

    if st.button("Submit Query"):
        with st.spinner("Fetching response..."):
            # Combine all text data
            all_texts = combine_text_data(transactions, social_media, customer_profiles, demographics)
            # Get top matches
            best_match = get_best_match(user_input, all_texts, model)
            # Display result
            st.subheader("Best Match:")
            st.write(best_match)



