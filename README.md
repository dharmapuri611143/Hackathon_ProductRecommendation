# AI-Powered Personalized Recommendation Engine

This project is a Streamlit application that generates personalized product and content recommendations based on customer profiles, transaction history, and social media activity using OpenAI's GPT-3.5-turbo model.

## Features

- **Adaptive Recommendations**: Suggests products or services based on recent shifts in customer behavior.
- **Personalized Suggestions**: Provides highly relevant product or service recommendations considering demographics, transactions, and social media interests.
- **Sentiment-Driven Content Recommendations**: Recommends content tailored to sentiment inferred from social media activity.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- Pandas
- ConfigParser
- LangChain
- OpenAI API Key

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    cd yourrepository
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Add your OpenAI API key to the `config.ini` file:
    ```ini
    [openai]
    api_key = your_openai_api_key
    ```

5. Ensure the following CSV files are in the project directory:
    - `customer_profiles.csv`
    - `transaction_history.csv`
    - `social_media_activity.csv`
    - `demographic_details.csv`

## Running the Application

To run the Streamlit application, execute the following command:
```sh
streamlit run hack_latest.py