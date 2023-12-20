from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.distance import distance as geopy_distance
from itertools import product
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import product
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from sentence_transformers import SentenceTransformer, util
import logging
import time
logging.basicConfig(filename='application.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from flask import Flask, request, jsonify
app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
from sentence_transformers import SentenceTransformer, util
domain_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(list1, list2):
    """
    Calculates cosine similarity between two lists of strings.

    Args:
    list1 (list of str): First list of strings.
    list2 (list of str): Second list of strings.

    Returns:
    float: Cosine similarity score between the two lists.
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(list1), ' '.join(list2)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        logging.info("Cosine similarity calculated successfully.")
        return similarity[0][0]
    except Exception as e:
        logging.error(f"Error in calculating cosine similarity: {e}")
        return 0

def location_similarity_median(list1, list2):
    """
    Calculates a similarity score based on the median geographical distance between locations in two lists.

    Args:
    list1 (list of str): First list of location names.
    list2 (list of str): Second list of location names.

    Returns:
    float: Normalized similarity score based on median geographical distance.
    """
    if not list1 or not list2:
        logging.warning("Empty location list(s) provided.")
        return 0

    geolocator = Nominatim(user_agent="geoapiExercises", timeout=10)
    distances = []

    for loc1, loc2 in product(list1, list2):
        try:
            time.sleep(1)  # Throttling request rate
            location1, location2 = geolocator.geocode(loc1), geolocator.geocode(loc2)
            if location1 and location2:
                distance = geopy_distance((location1.latitude, location1.longitude), (location2.latitude, location2.longitude)).kilometers
                distances.append(distance)
            else:
                logging.warning(f"Geocoding failed for locations: {loc1}, {loc2}")
        except Exception as e:
            logging.error(f"Error processing locations {loc1}, {loc2}: {e}")
            continue

    if distances:
        median_distance = np.median(distances)
        max_distance = 20000  # Approximate max distance on Earth in km
        return 1 - (median_distance / max_distance)
    else:
        logging.warning("No valid location pairs found for similarity calculation.")
        return 0   


def get_embedding(text):
    """
    Generates an embedding for the given text using a pre-trained BERT model.

    Args:
    text (str): The text to be embedded.

    Returns:
    numpy.ndarray: The embedding of the input text.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
    except Exception as e:
        logging.error(f"Error getting embedding for text '{text}': {e}")
        return None
    
def calculate_similarity(text1, text2):
    """
    Calculates the similarity between two text strings based on their embeddings.

    Args:
    text1 (str): First text string.
    text2 (str): Second text string.

    Returns:
    float: Similarity score between the two texts.
    """
    try:
        embedding1 = get_embedding(text1)
        embedding2 = get_embedding(text2)
        if embedding1 is not None and embedding2 is not None:
            return cosine_similarity(embedding1, embedding2)[0][0]
        else:
            return 0
    except Exception as e:
        logging.error(f"Error calculating similarity for texts: '{text1}', '{text2}': {e}")
        return 0


def median_similarity(list1, list2):
    """
    Computes the median of similarity scores between elements of two lists.

    Args:
    list1 (list): First list of elements.
    list2 (list): Second list of elements.

    Returns:
    float: Median similarity score between elements of the lists.
    """
    if not list1 or not list2:
        logging.warning("Empty list(s) provided for median similarity.")
        return 0

    similarities = []
    for item1, item2 in product(list1, list2):
        similarity = calculate_similarity(item1, item2)
        if similarity is not None:
            similarities.append(similarity)

    if similarities:
        return np.median(similarities)
    else:
        logging.warning("No valid pairs found for similarity calculation.")
        return 0

    
# def prepare_texts_for_lda(text_list):
#     if not text_list:
#         logging.warning("Empty text list provided for LDA preparation.")
#         return []
    
#     try:
#         tokens = [gensim.utils.simple_preprocess(text) for text in text_list]
#         return [item for sublist in tokens for item in sublist]
#     except Exception as e:
#         logging.error(f"Error in preparing texts for LDA. Error: {e}")
#         return []

# def get_topic_distribution(model, bow):
#     try:
#         topic_dist = model.get_document_topics(bow, minimum_probability=0)
#         return [prob for _, prob in topic_dist]
#     except Exception as e:
#         logging.error(f"Error in getting topic distribution. Error: {e}")
#         return []


# def industry_similarity(resume_industry_data, jd_industry_data):
#     try:
#         resume_tokens = prepare_texts_for_lda(resume_industry_data)
#         job_description_tokens = prepare_texts_for_lda(jd_industry_data)
#         dictionary = corpora.Dictionary([resume_tokens, job_description_tokens])
#         corpus = [dictionary.doc2bow(text) for text in [resume_tokens, job_description_tokens]]
        
#         lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, random_state=100, update_every=1, passes=10, alpha='auto')
        
#         resume_topic_dist = get_topic_distribution(lda_model, corpus[0])
#         job_desc_topic_dist = get_topic_distribution(lda_model, corpus[1])

#         if resume_topic_dist and job_desc_topic_dist:
#             return cosine_similarity([resume_topic_dist], [job_desc_topic_dist])[0][0]
#         else:
#             return 0
#     except Exception as e:
#         logging.error(f"Error in calculating industry similarity score. Error: {e}")
#         return 0
    
def industry_similarity(text1, text2):
    """
    Calculates semantic similarity between two industry-related text strings.

    Args:
    text1 (str): First industry-related text string.
    text2 (str): Second industry-related text string.

    Returns:
    float: Semantic similarity score between the two texts.
    """
    try:
        # Encoding the texts to get embeddings
        embedding1 = domain_model.encode(text1, convert_to_tensor=True)
        embedding2 = domain_model.encode(text2, convert_to_tensor=True)

        # Calculating cosine similarity between embeddings
        similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
        return similarity
    except Exception as e:
        # Logging the exception
        logging.error(f"Error in calculating industry similarity score for texts '{text1}' and '{text2}': {e}")
        return 0

@app.route('/calculate_cosine_similarity', methods=['POST'])
def calculate_cosine_similarity_endpoint():
    """
    Endpoint to calculate cosine similarity between two lists of strings.
    """
    try:
        data = request.json
        list1 = data.get('list1')
        list2 = data.get('list2')

        if not list1 or not list2:
            logging.warning("calculate_cosine_similarity: One or both lists are empty.")
            return jsonify({"error": "List1 or List2 is empty"}), 400

        similarity = calculate_cosine_similarity(list1, list2)
        return jsonify({"similarity": similarity}), 200
    except Exception as e:
        logging.error(f"Error in calculate_cosine_similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    

    

@app.route('/location_similarity', methods=['POST'])
def location_similarity_endpoint():
    """
    Endpoint to calculate geographical similarity between two lists of locations.
    """
    try:
        data = request.json
        list1 = data.get('list1')
        list2 = data.get('list2')

        if not list1 or not list2:
            logging.warning("location_similarity: One or both location lists are empty.")
            return jsonify({"error": "List1 or List2 is empty"}), 400

        similarity = location_similarity_median(list1, list2)
        return jsonify({"similarity": similarity}), 200
    except Exception as e:
        logging.error(f"Error in location_similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/text_similarity', methods=['POST'])
def text_similarity_endpoint():
    """
    Endpoint to calculate similarity between two text strings based on embeddings.
    """
    try:
        data = request.json
        text1 = data.get('text1')
        text2 = data.get('text2')

        if text1 is None or text2 is None:
            logging.warning("text_similarity: One or both texts are missing.")
            return jsonify({"error": "text1 or text2 is missing"}), 400

        similarity = calculate_similarity(text1, text2)
        return jsonify({"similarity": float(similarity)}), 200
    except Exception as e:
        logging.error(f"Error in text_similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/median_similarity', methods=['POST'])
def median_similarity_endpoint():
    """
    Endpoint to calculate the median similarity score between elements of two lists.
    """
    try:
        data = request.json
        list1 = data.get('list1')
        list2 = data.get('list2')

        if not list1 or not list2:
            logging.warning("median_similarity: One or both lists are empty.")
            return jsonify({"error": "List1 or List2 is empty"}), 400

        similarity = median_similarity(list1, list2)
        return jsonify({"similarity": float(similarity)}), 200
    except Exception as e:
        logging.error(f"Error in median_similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/industry_similarity', methods=['POST'])
def industry_similarity_endpoint():
    """
    Endpoint to calculate industry similarity between two text strings.
    """
    try:
        data = request.json
        text1 = data.get('text1')
        text2 = data.get('text2')

        if text1 is None or text2 is None:
            logging.warning("industry_similarity: One or both texts are missing.")
            return jsonify({"error": "text1 or text2 is missing"}), 400

        similarity = industry_similarity(text1, text2)
        return jsonify({"similarity": similarity}), 200
    except Exception as e:
        logging.error(f"Error in industry_similarity endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)   
    