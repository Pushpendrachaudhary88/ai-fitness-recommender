






# app.py (Final Version with Exercises and Improved UI)
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(page_title="FitRec: AI Recommender", page_icon="🏋️", layout="wide")
st.title("FitRec: AI-Powered Workout Recommender 🏋️‍♂️")

# --- File Path ---
# Nayi file ka naam istemal karein
FILE_PATH = "data/processed_workout_programs_with_exercises_reps.csv"

# --- Caching Functions ---
@st.cache_data
def load_and_process_data(path):
    """Loads data, handles errors, and creates the feature for the model."""
    try:
        df = pd.read_csv(path, engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        required_cols = ['title', 'description', 'level', 'goal', 'equipment', 'exercises']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"FATAL ERROR: Aapki CSV file mein yeh columns nahi mil rahe: {missing_cols}")
            st.stop()

        df.dropna(subset=required_cols, inplace=True)
        
        for col in required_cols:
            df[col] = df[col].astype(str)
        df['combined_text'] = df[required_cols].agg(' '.join, axis=1)
        
        return df

    except FileNotFoundError:
        st.error(f"FATAL ERROR: File not found at `{path}`.")
        st.info(f"Please sunishchit karein ki aapne sahi '{FILE_PATH}' file download karke isi folder mein rakhi hai.")
        st.stop()
    except Exception as e:
        st.error(f"Data load karte samay ek anjaan error aayi: {e}")
        st.stop()

@st.cache_resource
def build_model(_df):
    """Builds the TF-IDF vectorizer and matrix."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(_df['combined_text'])
    return vectorizer, tfidf_matrix

# --- Main App Logic ---
df = load_and_process_data(FILE_PATH)
vectorizer, tfidf_matrix = build_model(df)

# st.success("App safaltapoorvak load ho gaya hai! Ab aap recommender ka istemal kar sakte hain.")
st.success("Welcome To The Best AI Fitness Recommendation App.")

# --- UI Design ---
with st.sidebar:
    st.header("Apna Perfect Workout Dhoondhein")
    selected_program_title = st.selectbox(
        "Aapko kaunsa workout pasand hai?",
        options=sorted(df['title'].unique()),
        index=None,
        placeholder="Ek workout program chunein..."
    )
    top_n = st.slider("Kitne recommendations chahiye?", 5, 20, 10)
    st.markdown("---")
    st.info("Yeh project AI ka istemal karke aapke liye best workout dhoondhta hai.")

if selected_program_title:
    st.header(f"'{selected_program_title}' Jaise Aur Workout Programs:")

    if selected_program_title not in df['title'].values:
        st.error("Chuna gaya program nahi mil paaya.")
    else:
        idx = df[df['title'] == selected_program_title].index[0]
        query_vector = tfidf_matrix[idx]
        sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        similar_indices = sim_scores.argsort()[-top_n-1:-1][::-1]

        recommendations = df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = sim_scores[similar_indices]

        if recommendations.empty:
            st.warning("Iske jaise aur programs nahi mil paaye.")
        else:
            for i, rec in enumerate(recommendations.itertuples()):
                with st.container(border=True):
                    st.subheader(rec.title)
                    st.markdown(f"**Similarity Score:** `{rec.similarity_score:.2f}`")
                    
                    # Metrics Display
                    col1, col2 = st.columns(2)
                    col1.metric("Level", getattr(rec, 'level', 'N/A'))
                    col2.metric("Equipment", getattr(rec, 'equipment', 'N/A'))
                    
                    st.markdown("---")
                    
                    # Exercise List
                    st.markdown("**Sample Exercises (Sets x Reps):**")
                    exercises = getattr(rec, 'exercises', 'No exercises listed').split(', ')
                    for ex in exercises:
                        st.markdown(f"- {ex}")
                    
                    # Expander for details
                    with st.expander("Poora Description Dekhein"):
                        st.write(getattr(rec, 'description', 'No description available.'))
                st.write("") # Thoda space
else:
    st.info("Shuru karne ke liye, left sidebar se ek workout program chunein.")




