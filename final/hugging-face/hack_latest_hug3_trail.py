import streamlit as st
import pandas as pd
import configparser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from huggingface_hub import login
from langchain_community.llms import HuggingFaceHub
import os

# Read Hugging Face access token
config = configparser.ConfigParser()
config.read('config.ini')
hf_api_key = config['huggingface']['access_token']

# Hugging Face setup
login(hf_api_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# Load datasets
customer_profiles = pd.read_csv("customer_profiles.csv")
transactions = pd.read_csv("transaction_history.csv")
social_media = pd.read_csv("social_media_activity.csv")
demographics = pd.read_csv("demographic_details.csv")

base_template = """
Given the following detailed customer profile:

Demographic Information:
{demographics}

Recently Purchased Products:
{transactions}

Social Media Interests and Activities:
{social_media}

Please provide the following:
{category_prompt}

**Output Requirements:**
- Start each suggestion with a heading: **Recommendation:** followed by **Reason for Recommendation:** heading should not 
be 
repeated.
- Use bullet points under each outcome and reasoning.
- Highlight product names in **bold**.
- Provide specific, real-world **product or service names**.
- Include at least one **online source link (http URL)** for each product or content.
- Format everything using **Markdown** for clarity and readability.

PROMPT ENDED:
"""

prompt = PromptTemplate.from_template(base_template)

recommendation_chain = (
        {
            "demographics": RunnablePassthrough(),
            "transactions": RunnablePassthrough(),
            "social_media": RunnablePassthrough(),
            "category_prompt": RunnablePassthrough()
        } | prompt | llm | StrOutputParser()
)

st.set_page_config(page_title="✨ AI Personalized Recommendations ✨", layout="wide")
st.title("✨ AI-Powered Personalized Recommendation Engine")

tabs = st.tabs(["🔎 Existing Customer", "🆕 New Customer"])

categories = {
    "Adaptive Recommendation Engine": "1. Suggest a specific product or service that adapts to a recent shift in the customer's behavior with brand name and reasoning.",
    "AI-Generated Personalized Product/Service Suggestions": "2. Recommend at least 2 highly relevant products/services with examples and reasoning.",
    "Sentiment-Driven Content Recommendations": "3. Recommend content (videos, blogs, etc.) based on sentiment analysis of social media posts. Note: Prompt should not be included in the response",
    "Predictive Customer Insights & Business Strategies": "4. Predict churn risks and suggest retention strategies with examples.",
    "Multi-Modal Personalization": "5. Use text, image, or voice input to personalize suggestions with examples. Act like a chatbot to respond to the user interactively using provided multi-modal input.",
    "Hyper-Personalized Financial Product Recommendations": "6. Suggest credit plans, loans, or investments based on transaction and risk profile. Note: Prompt should not be included in the response"
}

# Existing Customer Tab (unchanged)
with tabs[0]:
    st.header("🔍 Existing Customer Profile")
    customer_id = st.selectbox("Select Customer ID:", customer_profiles['Customer_Id'].unique())

    demo = demographics[demographics['Customer_Id'] == customer_id].iloc[0]
    customer_trans = transactions[transactions['Customer_Id'] == customer_id][['Purchase_Date', 'Category', 'Amount (In Dollars)']]
    customer_social = social_media[social_media['Customer_Id'] == customer_id][['Timestamp', 'Content']]

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

    option = st.selectbox("🚀 Choose Recommendation:", list(categories.keys()), key="existing_category")

    if option == "Multi-Modal Personalization":
        st.subheader("🧠 Chatbot Interaction Mode")
        user_text = st.text_area("💬 Your financial goals or lifestyle:", key="existing_multimodal")

        if st.button("🤖 Generate Multi-Modal Response"):
            if user_text.strip() == "":
                st.warning("Please enter your financial goals or lifestyle first.")
            else:
                with st.spinner("Generating..."):
                    response = recommendation_chain.invoke({
                        "demographics": demo.drop("Customer_Id").to_dict(),
                        "transactions": ", ".join(customer_trans['Category'].tolist()),
                        "social_media": ", ".join(customer_social['Content'].tolist()),
                        "category_prompt": f"Use input '{user_text}' to respond interactively. {categories[option]}"
                    })
                st.success("✅ Response Generated!")
                st.markdown(response.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)

    else:
        if st.button("🚀 Generate Recommendation"):
            with st.spinner("Generating recommendation..."):
                recommendations = recommendation_chain.invoke({
                    "demographics": demo.drop("Customer_Id").to_dict(),
                    "transactions": ", ".join(customer_trans['Category'].tolist()),
                    "social_media": ", ".join(customer_social['Content'].tolist()),
                    "category_prompt": categories[option]
                })
            st.success("✅ Recommendation Generated!")
            st.markdown(recommendations.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)

# New Customer Tab (added)
with tabs[1]:
    st.header("✨ New Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        demo_input = st.text_area("📋 Demographics (JSON format)", json.dumps({
            "Location": "Suburban",
            "Marital Status": "Married",
            "Dependents": 2,
            "Home Ownership": "With Family",
            "Nationality": "Indian"
        }, indent=4), height=200)

    with col2:
        trans_input = st.text_area("🛒 Transactions (comma-separated)", "Gucci, Mutual Funds, Supermarket")
        social_input = st.text_area("📲 Social Media (comma-separated)", "Posts about financial management concerns.")

    try:
        demo_dict = json.loads(demo_input)
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format in demographics!")
        st.stop()

    option_new = st.selectbox("🚀 Choose Recommendation:", list(categories.keys()), key="new_category")

    if st.button("🚀 Generate New Customer Recommendation"):
        with st.spinner("Generating recommendation..."):
            recommendations_new = recommendation_chain.invoke({
                "demographics": demo_dict,
                "transactions": trans_input,
                "social_media": social_input,
                "category_prompt": categories[option_new]
            })
        st.success("✅ Recommendation Generated!")
        st.markdown(recommendations_new.split("PROMPT ENDED:")[1].strip(), unsafe_allow_html=True)