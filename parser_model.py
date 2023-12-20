from tika import parser
import os
import spacy
import pandas as pd
from spacy.matcher import Matcher
from nltk.corpus import stopwords
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_lg")
from resume_parser import resumeparse
from pyresparser import ResumeParser
matcher = Matcher(nlp.vocab)
from config import *
import re
from nltk.stem import WordNetLemmatizer
import tempfile
# import PyPDF2
import logging
from flask import Flask, request, jsonify
import tempfile


app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
skills_df = pd.read_csv(skills_file_path)
# def process_pdf(file_stream):
#     # Read PDF file from file stream
#     pdf_reader = PyPDF2.PdfReader(file_stream)
#     text = ''
#     text = ''
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# def extract_pdf(file):
#     if file.filename.lower().endswith('.pdf'):
#         return process_pdf(file)
#     else:
#         raise ValueError("Unsupported file type")

# def get_email_addresses(string):
#     r = re.compile(r'[\w\.-]+@[\w\.-]+')
#     return r.findall(string)

# def get_phone_numbers(string):
#     r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
#     phone_numbers = r.findall(string)
#     return [re.sub(r'\D', '', num) for num in phone_numbers]

# def extract_name(resume_text):
#     nlp_text = nlp(resume_text)
    
#     # First name and Last name are always Proper Nouns
#     pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    
#     matcher.add('NAME', [pattern], on_match = None)
    
#     matches = matcher(nlp_text)
    
#     for match_id, start, end in matches:
#         span = nlp_text[start:end]
#         return span.text

def save_temp_file(file):
    """
    Saves an uploaded file to a temporary file.

    Args:
    file (FileStorage): The file uploaded by the user.

    Returns:
    str: The path to the temporary file.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            logging.info(f"Temporary file saved at {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logging.error(f"Error in saving temporary file: {e}")
        return None

def clean_text(text):
    """
    Cleans the input text by lowercasing, removing special characters, new lines, 
    and stopwords, and performs lemmatization.

    Args:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    try:
        # Lowercasing
        text = text.lower()

        # Removing new lines and special characters
        text = re.sub(r'\W', ' ', text.replace("\n", " "))

        # Tokenization
        tokens = text.split()

        # Removing stopwords and lemmatization
        lemmatizer = WordNetLemmatizer()
        cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

        # Joining back into string
        cleaned_text = ' '.join(cleaned_tokens)
        return cleaned_text
    except Exception as e:
        logging.error(f"Error in cleaning text: {e}")
        return None

def combine_lists(list1, list2):
    """
    Combines two lists into one, removing duplicates.

    Args:
    list1 (list): The first list or None.
    list2 (list): The second list or None.

    Returns:
    list: A combined list with unique elements.
    """
    try:
        if not isinstance(list1, (list, type(None))) or not isinstance(list2, (list, type(None))):
            raise TypeError("Input arguments must be lists or None")

        list1 = list1 if list1 is not None else []
        list2 = list2 if list2 is not None else []

        if list1 or list2:
            combined_list = set(list1) | set(list2)
            return list(combined_list)
        else:
            return []
    except TypeError as e:
        logging.error(f"TypeError in combine_lists: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in combine_lists: {e}")
        return []    
# def clean_text_data(text):
#     ps = PorterStemmer()
#     wordnet=WordNetLemmatizer()
#     sentences = nltk.sent_tokenize(text)
#     corpus = []
#     text = ''
#     for i in range(len(sentences)):
#         review = re.sub('[^a-zA-Z]', ' ', sentences[i])
#         review = review.lower()
#         review = review.split()
#         review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
#         review = ' '.join(review)
#         text += ' ' + review
#     return review
    
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     text = text.replace('\n', ' ').replace('\t', ' ')
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
    
def extract_locations(text):
    """
    Extracts location names from a text string using NLP.

    Args:
    text (str): The text from which to extract locations.

    Returns:
    list: A list of extracted location names.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        cleaned_text = re.sub(r'\W', ' ', text.lower())
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()

        doc = nlp(cleaned_text)
        locations_list = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        return locations_list
    except TypeError as e:
        logging.error(f"TypeError in extract_locations: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in extract_locations: {e}")
        return []
    

def extract_file_data(file):
    """
    Extracts and aggregates data from a given file, typically a resume or a job description.

    Args:
    file (FileStorage): A file uploaded by the user.

    Returns:
    dict: A dictionary containing extracted data like skills, universities, designations, degrees, and locations.
    """
    try:
        logging.info("Starting file data extraction.")

        # Saving the uploaded file temporarily
        temp_file_path = save_temp_file(file)
        if not temp_file_path:
            logging.error("Failed to save the uploaded file temporarily.")
            return {}

        # Parsing the file to extract text content
        file_data = parser.from_file(temp_file_path)
        text = file_data.get('content', '')
        if not text:
            logging.warning("No content extracted from the file.")
            return {}

        cleaned_text = clean_text(text)
        logging.info("File parsed and text cleaned successfully.")

        # Using different parsers to extract data
        resume_parser_first = ResumeParser(temp_file_path, skills_file=skills_file_path).get_extracted_data()
        logging.info("Data extracted using the first parser.")
        
        resume_parser_second = resumeparse.read_file(temp_file_path)
        logging.info("Data extracted using the second parser.")

        # Combining extracted data from both parsers
        extracted_data = {
            'skills': combine_lists(resume_parser_second.get('skills'), resume_parser_first.get('skills')),
            'university': combine_lists(resume_parser_second.get('university'), resume_parser_first.get('college_name')),
            'designations': combine_lists(resume_parser_first.get('designation'), resume_parser_second.get('designition')),
            'education_degree': combine_lists(resume_parser_first.get('degree'), resume_parser_second.get('degree')),
            'locations': extract_locations(text),
            'description': cleaned_text
        }

        logging.info("Data combined and aggregated successfully.")
        
        # Clean up: Removing the temporary file
        os.remove(temp_file_path)
        return extracted_data

    except Exception as e:
        logging.error(f"An error occurred during file data extraction: {e}")
        # Attempt to clean up the temporary file in case of an error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {}
    

@app.route('/parse-pdf', methods=['POST'])
def parse_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            extracted_data = extract_file_data(file)
            return jsonify(extracted_data), 200
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


def allowed_file(filename):
    # Check if the file is a PDF (you can modify this to allow other file types)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

# Adapt your existing extract_file_data function to handle the Flask file object.
# For example, you might need to save the file temporarily and pass its path to your parsing functions.

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    



        

