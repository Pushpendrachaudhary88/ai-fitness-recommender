# src/recommender.py (FINAL Corrected Code)
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_and_process_data(path: str) -> pd.DataFrame:
    """Loads, cleans, and engineers features in one go."""
    # Step 1: Load data, skipping any bad lines
    df = pd.read_csv(path, engine='python', on_bad_lines='skip')
    
    # Step 2: Handle missing values in essential columns
    required_cols = ['title','description','level','goal','equipment']
    df.dropna(subset=required_cols, inplace=True)
    
    # Step 3: Create the 'combined_text' feature
    # Ensure all columns are strings before joining
    for col in required_cols:
        df[col] = df[col].astype(str)
        
    df['combined_text'] = df[required_cols].agg(' '.join, axis=1)
    
    # Step 4: Final cleaning
    df.dropna(subset=['combined_text'], inplace=True)
    df['title'] = df['title'].str.strip()
    
    return df

@st.cache_resource
def build_model(df: pd.DataFrame):
    """Builds the TF-IDF vectorizer and matrix."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    return vectorizer, tfidf_matrix

def get_recommendations(df, vectorizer, tfidf_matrix, query_title, top_n=10):
    """Finds workouts similar to the query_title."""
    if query_title not in df['title'].values:
        return pd.DataFrame()
    
    idx = df[df['title'] == query_title].index[0]
    query_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    similar_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    
    recommendations = df.iloc[similar_indices].copy()
    recommendations['similarity_score'] = sim_scores[similar_indices]
    
    return recommendations

