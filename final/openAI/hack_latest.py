import streamlit as st
import pandas as pd
import configparser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

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

tabs = st.tabs(["🔎 Existing Customer", "🆕 New Customer"])

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
        demo_input = st.text_area("Demographics (JSON format)", height=200, value=json.dumps({
            "Customer_Id": 4001,
            "Location": "Suburban",
            "Marital Status": "Married",
            "Dependents": 2,
            "Home Ownership": "Living with Family",
            "Nationality": "Indian"
        }, indent=4))

    with col2:
        st.subheader("🛒 Recent Transactions & 📲 Social Media")
        trans_input = st.text_area("Recent Transactions (comma-separated)", "Gucci, Mutual Funds, Supermarket")
        social_input = st.text_area("Social Media Activities (comma-separated)", "Sample post content related to Financial Management Concern.")

    if st.button("🚀 Generate Recommendations", key="new"):
        try:
            demo_dict = json.loads(demo_input)
        except json.JSONDecodeError:
            st.error("Demographic info JSON is invalid. Please correct it.")
            st.stop()

        with st.spinner("Generating personalized recommendations..."):
            recommendations_new = recommendation_chain.invoke({
                "demographics": demo_dict,
                "transactions": trans_input,
                "social_media": social_input
            })
        st.success("✅ Recommendations Generated for New Customer!")
        st.subheader("🎯 Personalized Recommendations")
        st.markdown(recommendations_new, unsafe_allow_html=True)
