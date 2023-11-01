import os
import re
import json
import streamlit as st

def extract_list_items(string_with_enumerate):
    items = string_with_enumerate.split("\n")
    extracted_list = []

    for item in items:
        index = item.find(". ")
        if index != -1:
            extracted_list.append(item[index + 2:])

    return extracted_list

def init_environment():
    os.environ["OPENAI_API_KEY"] = st.secrets["open_ai_api_key"]
    os.environ["REPLICATE_API_TOKEN"] = st.secrets["replicate"]


def extract_json_from_text(text) -> dict:
    # Use regular expression to match content between backticks
    match = re.search(r'```(.*?)```', text, re.DOTALL)

    # If a match is found, attempt to parse the content as JSON
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Extracted content is not valid JSON")
    else:
        raise ValueError("No JSON content found between backticks")
