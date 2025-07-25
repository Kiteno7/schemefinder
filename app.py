# app.py
import sqlite3
import json
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ---- Initialize the Flask App ----
app = Flask(__name__, template_folder='.')

# ---- Database Setup ----
def get_db_connection():
    conn = sqlite3.connect('schemes.db')
    conn.row_factory = sqlite3.Row
    return conn

# Load all schemes into a Pandas DataFrame for easier processing
conn = get_db_connection()
all_schemes_df = pd.read_sql_query("SELECT * FROM schemes", conn)
conn.close()

# Combine relevant text fields for the TF-IDF model
all_schemes_df['search_text'] = all_schemes_df['name'] + ' ' + all_schemes_df['description'] + ' ' + all_schemes_df['eligibility_summary'] + ' ' + all_schemes_df['keywords']

# ---- AI/NLP Model Setup ----
# Create a TF-IDF Vectorizer
# This will convert our text into a matrix of numbers that our model can understand.
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the scheme data
# 'fit_transform' learns the vocabulary and creates the matrix for our schemes.
tfidf_matrix = vectorizer.fit_transform(all_schemes_df['search_text'])


# ---- API Endpoint for Searching ----
@app.route('/api/search', methods=['POST'])
def search():
    # Get the user's query from the request
    query_data = request.get_json()
    user_query = query_data.get('query', '')

    if not user_query:
        return jsonify([])

    # Transform the user's query using the same vectorizer
    # 'transform' uses the vocabulary learned from our schemes.
    query_vector = vectorizer.transform([user_query])

    # Calculate cosine similarity between user query and all schemes
    # This gives a score of how similar the query is to each scheme.
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the top 3 most similar schemes
    # We set a threshold of 0.1 to avoid showing completely irrelevant results.
    relevant_indices = [i for i, score in enumerate(cosine_similarities) if score > 0.1]
    
    # Sort by similarity score in descending order and take the top 5
    top_indices = sorted(relevant_indices, key=lambda i: cosine_similarities[i], reverse=True)[:5]

    # Get the results from the DataFrame
    results = all_schemes_df.iloc[top_indices]

    # Convert results to a list of dictionaries to send as JSON
    return jsonify(results.to_dict(orient='records'))


# ---- Route to serve the main HTML page ----
@app.route('/')
def index():
    return render_template('index.html')


# ---- Run the App ----
if __name__ == '__main__':
    # Using port 8000 for local development
    app.run(host='0.0.0.0', port=8000, debug=True)