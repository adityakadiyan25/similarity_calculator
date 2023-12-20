from flask import Flask, request, jsonify
from parser_model import *
from score_calculator import *
import logging
from config import *

import os

ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
logging.basicConfig(filename='application.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_final_similarity_score(resume_extracted_data, jd_extracted_data):
    """
    Calculates the final similarity score between a resume and a job description.

    Args:
    resume_extracted_data (dict): Extracted data from the resume.
    jd_extracted_data (dict): Extracted data from the job description.

    Returns:
    float or None: The final similarity score, or None in case of an error.
    """
    try:
        # Calculating individual similarity scores
        skills_similarity_score = calculate_cosine_similarity(resume_extracted_data['skills'], jd_extracted_data['skills'])
        location_similarity_score = location_similarity_median(resume_extracted_data['locations'], jd_extracted_data['locations'])
        median_designation_similarity = median_similarity(resume_extracted_data['designations'], jd_extracted_data['designations'])
        median_degree_similarity = median_similarity(resume_extracted_data['education_degree'], jd_extracted_data['education_degree'])
        industry_similarity_score = industry_similarity(resume_extracted_data['description'], jd_extracted_data['description'])

        # Aggregating individual scores into a final score
        final_score = ((skill_weightage * skills_similarity_score) + 
                       (location_weightage * location_similarity_score) + 
                       (designation_weightage * median_designation_similarity) + 
                       (degree_weightage * median_degree_similarity) + 
                       (industry_weightage * industry_similarity_score)) * 100

        logging.info("Final similarity score calculated successfully.")
        return final_score
    except Exception as e:
        logging.error(f"Error in calculate_final_similarity_score: {e}")
        return None

def allowed_file(filename):
    """
    Checks if the filename has an allowed extension.

    Args:
    filename (str): The name of the file to check.

    Returns:
    bool: True if the file has an allowed extension, False otherwise.
    """
    try:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    except Exception as e:
        logging.error(f"Error in allowed_file: {e}")
        return False

    
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    """
    Flask endpoint to calculate the similarity between a resume and a job description.
    """
    try:
        if 'resume' not in request.files or 'job_description' not in request.files:
            logging.warning("calculate_similarity endpoint: Missing file part.")
            return jsonify({"error": "No resume or job description file part"}), 400

        resume_file = request.files['resume']
        job_description_file = request.files['job_description']

        if resume_file.filename == '' or job_description_file.filename == '':
            logging.warning("calculate_similarity endpoint: No file selected.")
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(resume_file.filename) or not allowed_file(job_description_file.filename):
            logging.warning("calculate_similarity endpoint: Invalid file type.")
            return jsonify({"error": "One or both files are not allowed"}), 400

        resume_data = extract_file_data(resume_file)
        job_description_data = extract_file_data(job_description_file)
        final_similarity_score = calculate_final_similarity_score(resume_data, job_description_data)

        return jsonify({"message": "File processed", "final_similarity_score": final_similarity_score}), 200
    except Exception as e:
        logging.error(f"Error in calculate_similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001 , debug=True) 