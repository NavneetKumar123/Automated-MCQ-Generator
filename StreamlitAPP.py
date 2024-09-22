import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file, get_table_data

# Loading JSON file
with open(r'C:\Users\Asus\Downloads\Automated-MCQ-Generator-Using-Langchain-OpenAI-API-main\Automated-MCQ-Generator-Using-Langchain-OpenAI-API-main\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Creating a title for the app
st.title("MCQs Generator Application with Langchain")

# Create a form using st.form
with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or Text file",type=['pdf', 'txt'])
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level of questions", max_chars=20, placeholder="simple")
    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading..."):
            try:
                # Reading the file
                text = read_file(uploaded_file)
                
                # Get OpenAI callback and generate the response
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(RESPONSE_JSON)
                        }
                    )
                    
                    # Log token usage
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion Tokens: {cb.completion_tokens}")
                    print(f"Total Cost: {cb.total_cost}")

            except Exception as e:
                # Print and log the full exception
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error(f"Error: {str(e)}")
            else:
                # If response is a dictionary, extract and display the quiz data
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in the table generation.")
                else:
                    st.write(response)
