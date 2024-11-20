import streamlit as st
from streamlit import session_state as ss
from io import StringIO
import pdfplumber
import pandas as pd
import regex as re  # Use the `regex` module for advanced regular expressions
from streamlit_pdf_viewer import pdf_viewer
from openai import OpenAI
# from PyPDF2 import PdfReader
import base64
import random
import json
import importlib
import gmft
import gmft.table_detection
import gmft.table_visualization
import gmft.table_function
import gmft.table_function_algorithm
import gmft.table_captioning
import gmft.pdf_bindings.bindings_pdfium
import gmft.pdf_bindings
import gmft.common
import pandas as pd

importlib.reload(gmft)
importlib.reload(gmft.common)
importlib.reload(gmft.table_captioning)
importlib.reload(gmft.table_detection)
importlib.reload(gmft.table_visualization)
importlib.reload(gmft.table_function)
importlib.reload(gmft.table_function_algorithm)
importlib.reload(gmft.pdf_bindings.bindings_pdfium)
importlib.reload(gmft.pdf_bindings)

from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import CroppedTable, AutoTableDetector

detector = AutoTableDetector()

from gmft.auto import AutoTableFormatter

formatter = AutoTableFormatter()

from gmft.auto import AutoFormatConfig


config_hdr = AutoFormatConfig() # config may be passed like so
config_hdr.verbosity = 3
config_hdr.enable_multi_header = True
config_hdr.semantic_spanning_cells = True # [Experimental] Merge headers

api_key = st.secrets["openai_api_key"]

# Pass the API key directly when creating the client
client = OpenAI(api_key=api_key)

# Mock FORMAT and Client setup
FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "yes_no_response",
        "schema": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "enum": ["Yes", "No"]
                }
            },
            "required": ["response"],
            "additionalProperties": False
        },
        "strict": True
    }
}


# Dummy function for chat API call (replace with your actual API integration)
def is_table_valid(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format = FORMAT,
        max_tokens=200,
        temperature=0
    )
    output = response.choices[0].message.content.strip()
    output_dict = json.loads(output)
    return output_dict


# PDF ingestion logic
def ingest_pdf(pdf_path):
    doc = PyPDFium2Document(pdf_path)
    tables = []
    for page in doc:
        tables += detector.extract(page)
    return tables, doc


def ingest_pdf_first_page(pdf_path):
    doc = PyPDFium2Document(pdf_path)
    tables = []
    for page in doc:
        page_tables = detector.extract(page)
        if page_tables:  # If tables are found on the page
            tables.append(page_tables[0])  # Add the first table from the page
            break  # Stop after extracting the first table
    return tables, doc


def clean_text(text):
    if isinstance(text, str):
        # Remove double quotes
        text = text.replace('"', '')
        # Remove escape sequences by encoding and decoding
        text = text.replace('\n', ' ').replace('\t', ' ')
    return text

# Streamlit app
def pdf_viewer(input, width=700, height=800):
    # Convert binary data to base64 for embedding
    base64_pdf = base64.b64encode(input).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" style="border:none;"></iframe>'
    st.components.v1.html(pdf_display, height=height + 50)

def premium_interface():
    st.title("PDF Table Extractor (Premium)")

    # File uploader
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Process PDF if uploaded
    if pdf_file is not None:
        binary_data = pdf_file.getvalue()
        st.write("Preview of the uploaded PDF:")
        pdf_viewer(input=binary_data)

        with pdfplumber.open(pdf_file) as pdf:
            num_pages = len(pdf.pages)
            st.write(f"The PDF has {num_pages} pages.")

            # Input for page numbers
            starting_page = st.number_input("Enter the starting page number:", min_value=1, max_value=num_pages,
                                            value=1)
            ending_page = st.number_input("Enter the ending page number:", min_value=starting_page, max_value=num_pages,
                                          value=num_pages)

            extracted_text = ""
            # Extract text from selected pages
            # if st.button("Extract Text from Selected Pages"):
            for i in range(starting_page - 1, ending_page):  # Convert to zero-index
                if i < len(pdf.pages):  # Ensure the page exists
                    page = pdf.pages[i]
                    extracted_text += page.extract_text() + "\n"
                # st.text_area("Extracted Text:", extracted_text, height=300)

            # Input for column names and exactly 3 rows
            st.write("Enter column names and rows (2 rows are atleast mandatory) for the model to understand the schema:")
            column_names = st.text_input('Enter column names (Eg: "Student Name","Course Title","Instructor":')
            rows = []
            for i in range(3):
                if i < 2:
                    prompt = f'Enter mandatory row {i + 1} (e.g., "John Doe","Introduction to Computer Science","Dr. Emily Carter"):'
                else:
                    prompt = f'Enter optional row {i + 1} (e.g., "Jane Smith","Physics 101","Dr. Sarah Johnson"):'

                row = st.text_input(prompt)
                if row:  # Only append non-empty rows
                    rows.append(row)

            # Generate the final prompt and display
            if st.button("Generate Table Output"):
                # Combine column names and rows into a single table
                if column_names and len(rows) >= 1:
                    table_input = column_names + "\n" + "\n".join(rows)

                    # Prompt for API
                    prompt = f"""
                    Please extract the table(s) as csv from this PDF. Ensure that each field is transcribed word-for-word exactly as it appears in the document without adding or modifying any information. Keep the original order of rows and columns and include every row on the first page. Please ensure that all values, even blank cells, match the original document exactly. Please do not miss any rows in the first page of the document.
                    Do not include 'Certainly! Below is the extracted data from the first page of the PDF'. Answer should be directly readable as csv.
                    Text:
                    {extracted_text}
                    """
                    example_prompt = f"""
                    Input:
                    {extracted_text[:500]}

                    Output:
                    {table_input}
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system",
                             "content": "You are a helpful assistant that generates csv text from pdf"},
                            {"role": "system", "content": example_prompt},
                            {"role": "user", "content": prompt},
                        ]
                    )

                    # Step 4: Print response
                    table_output = response.choices[0].message.content
                    data = StringIO(table_output)
                    df = pd.read_csv(data, on_bad_lines='skip')
                    df = df.apply(lambda col: col.map(clean_text))
                    st.dataframe(df)
                    csv_file = "extracted_table.csv"
                    df.to_csv(csv_file, index=False)

                    # Provide download button
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            label="Download CSV",
                            data=f,
                            file_name="extracted_table.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("Please ensure you have entered column names and exactly 3 rows.")

def main_interface():
    st.title("PDF Table Extractor")
    # st.write("Upload a PDF to extract and display tables.")

    # File uploader
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Process PDF if uploaded
    if pdf_file is not None:
        binary_data = pdf_file.getvalue()
        st.write("Preview of the uploaded PDF:")
        pdf_viewer(input=binary_data)

    # Process PDF if uploaded
    if pdf_file:
        st.write("Processing PDF...")
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())

        # Extract first table and validate
        tables, doc = ingest_pdf_first_page("temp.pdf")
        if tables:
            ft = formatter.extract(tables[0])
            csv_data = ft.df(config_overrides=config_hdr)  # Customize config as needed
            # print(csv_data)

            if csv_data.columns.duplicated().any():
                csv_data.columns = [f"Column_{i}" for i in range(csv_data.shape[1])]

            rows_to_send = csv_data.head(2).to_dict(orient="records")
            prompt = f"""
            The following data was extracted from a PDF file:

            First Two Rows:
            {rows_to_send}

            Does this data look properly extracted and structured? Data can have hierarchical header
            Just respond with Yes/No
            """
            output = is_table_valid(prompt)

            if output["response"] == "Yes":
                # Extract all tables and display
                st.write("Tables look valid. Extracting all tables...")
                all_tables, doc = ingest_pdf("temp.pdf")
                for i, table in enumerate(all_tables):
                    ft = formatter.extract(table)
                    table_data = ft.df(config_overrides=config_hdr)  # Customize config as needed

                    # Remove columns where all values are None or NaN
                    table_data = table_data.dropna(axis=1, how="all")

                    # Handle duplicate column names
                    if table_data.columns.duplicated().any():
                        table_data.columns = [
                            f"{col}_{idx}" if dup else col
                            for idx, (col, dup) in
                            enumerate(zip(table_data.columns, table_data.columns.duplicated(keep=False)))
                        ]

                    # def clean_text(text):
                    #     if isinstance(text, str):
                    #         # Remove double quotes
                    #         # text = text.replace('"', '')
                    #         # Remove escape sequences by encoding and decoding
                    #         text = text.replace('\n', ' ').replace('\t', ' ')
                    #     return text

                    # Apply the function to the entire DataFrame
                    table_data = table_data.apply(lambda col: col.map(clean_text))

                    # Write the table to Streamlit
                    st.write(f"Table {i + 1}")
                    st.dataframe(table_data)
                    csv_file = "extracted_table.csv"
                    table_data.to_csv(csv_file, index=False)

                    # Provide download button
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            label="Download CSV",
                            data=f,
                            file_name="extracted_table.csv",
                            mime="text/csv",
                            key=f"{i}"
                        )
            else:
                st.warning("The PDF is complicated. Use the premium version.")
        else:
            st.error("No tables found in the PDF.")

def main():
    # Check if the session state is set
    if "premium" not in st.session_state:
        st.session_state.premium = False

    # Sidebar to toggle Premium Version
    with st.sidebar:
        st.title("Navigation")
        if st.button("Try Premium Version"):
            st.session_state.premium = True

    # Toggle interface
    if st.session_state.premium:
        premium_interface()
    else:
        main_interface()

if __name__ == "__main__":
    main()
