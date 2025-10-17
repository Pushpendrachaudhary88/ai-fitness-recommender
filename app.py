# # app.py (FINAL Corrected Code)
# import streamlit as st
# # Naye function ko import karein
# from src.recommender import load_and_process_data, build_model, get_recommendations

# # --- Page Configuration ---
# st.set_page_config(page_title="FitRec: AI Workout Recommender", page_icon="🏋️", layout="wide")

# # --- Data and Model Loading ---
# FILE_PATH = "data/processed_workout_programs.csv"  # Correct path, no space

# try:
#     # Naye function ko call karein
#     df = load_and_process_data(FILE_PATH)
#     vectorizer, tfidf_matrix = build_model(df)
# except FileNotFoundError:
#     st.error(f"ERROR: File not found at `{FILE_PATH}`. Please ensure the file is in the 'data' folder.")
#     st.stop()
# except Exception as e:
#     st.error(f"An unexpected error occurred: {e}")
#     st.stop()

# # --- UI Design ---
# st.title("FitRec: AI-Powered Workout Recommender 🏋️‍♂️")
# st.write("Aapke liye perfect workout program, aapke goals ke anusaar.")

# with st.sidebar:
#     st.header("Apne Liye Perfect Workout Dhoondhein")
#     selected_program_title = st.selectbox(
#         "Ek workout program chunein jiske jaise aur program aap dekhna chahte hain:",
#         options=df['title'].unique(),
#         index=None,
#         placeholder="Ek workout program select karein..."
#     )
#     top_n = st.slider("Kitne recommendations dekhna chahte hain?", 5, 20, 10)
#     st.markdown("---")
#     st.info("Yeh project Content-Based Filtering, TF-IDF, aur Cosine Similarity ka istemal karta hai.")

# if selected_program_title:
#     st.header(f"'{selected_program_title}' Jaise Aur Workout Programs:")
#     recommendations = get_recommendations(df, vectorizer, tfidf_matrix, selected_program_title, top_n)
    
#     if recommendations.empty:
#         st.warning("Iske jaise aur programs nahi mil paaye.")
#     else:
#         num_cols = 3
#         cols = st.columns(num_cols)
#         for i, rec in enumerate(recommendations.itertuples()):
#             with cols[i % num_cols]:
#                 with st.container(border=True):
#                     st.subheader(rec.title)
#                     st.markdown(f"**Similarity Score:** `{rec.similarity_score:.2f}`")
#                     with st.expander("Aur Jaankari Dekhein"):
#                         st.markdown(f"**Goal:** {getattr(rec, 'goal', 'N/A')}")
#                         st.markdown(f"**Level:** {getattr(rec, 'level', 'N/A')}")
#                         st.markdown(f"**Equipment:** {getattr(rec, 'equipment', 'N/A')}")
#                         st.write(getattr(rec, 'description', 'No description available.'))
# else:
#     st.info("Shuru karne ke liye, left sidebar se ek workout program chunein.")





# # app.py (Final, All-in-One, Guaranteed Code)
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # --- Page Configuration ---
# st.set_page_config(page_title="FitRec: AI Recommender", page_icon="🏋️", layout="wide")
# st.title("FitRec: AI-Powered Workout Recommender 🏋️‍♂️")

# # --- File Path ---
# FILE_PATH = "data/processed_workout_programs.csv"

# # --- Caching Functions ---
# # Yeh functions data aur model ko ek hi baar load karte hain, jisse app fast chalta hai.
# @st.cache_data
# def load_and_process_data(path):
#     """Loads data, handles errors, and creates the feature for the model."""
#     try:
#         # File ko robust tareeke se padhein
#         df = pd.read_csv(path, engine='python', on_bad_lines='skip')

#         # === CRITICAL FIX: Column Names ko Saaf Karein ===
#         # Kabhi-kabhi column names mein extra space aa jaata hai, usse hatayein.
#         df.columns = df.columns.str.strip()

#         # Check karein ki zaroori columns hain ya nahi
#         required_cols = ['title', 'description', 'level', 'goal', 'equipment']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             st.error(f"FATAL ERROR: Aapki CSV file mein yeh columns nahi mil rahe: {missing_cols}")
#             st.info("Please yahan se download ki hui 'processed_workout_programs.csv' file hi istemal karein.")
#             st.stop()

#         # Zaroori columns mein khaali rows ko hatayein
#         df.dropna(subset=required_cols, inplace=True)
        
#         # AI model ke liye 'combined_text' feature banayein
#         for col in required_cols:
#             df[col] = df[col].astype(str)
#         df['combined_text'] = df[required_cols].agg(' '.join, axis=1)
        
#         return df

#     except FileNotFoundError:
#         st.error(f"FATAL ERROR: File not found at `{path}`.")
#         st.info("Please sunishchit karein ki `processed_workout_programs.csv` file `data` folder ke andar hi rakhi hai.")
#         st.stop()
#     except Exception as e:
#         # Yeh aapka error dikha raha hai
#         st.error(f"An unexpected error occurred during data loading: {e}")
#         st.stop()

# @st.cache_resource
# def build_model(_df):
#     """Builds the TF-IDF vectorizer and matrix."""
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
#     tfidf_matrix = vectorizer.fit_transform(_df['combined_text'])
#     return vectorizer, tfidf_matrix

# # --- Main App Logic ---

# # 1. Data Load aur Process Karein
# df = load_and_process_data(FILE_PATH)

# # 2. AI Model Banayein
# vectorizer, tfidf_matrix = build_model(df)


# # st.success("App safaltapoorvak load ho gaya hai! Ab aap recommender ka istemal kar sakte hain.")
# st.success("Welcome to the best AI Fitness Recommendation App. ")
# # --- UI Design ---
# with st.sidebar:
#     st.header("Apna Perfect Workout Dhoondhein")
#     selected_program_title = st.selectbox(
#         "Aapko kaunsa workout pasand hai?",
#         options=df['title'].unique(),
#         index=None,
#         placeholder="Ek workout program chunein..."
#     )
#     top_n = st.slider("Kitne recommendations chahiye?", 5, 20, 10)
#     st.markdown("---")
#     st.info("Yeh project AI (TF-IDF & Cosine Similarity) ka istemal karke aapke liye best workout dhoondhta hai.")


# if selected_program_title:
#     st.header(f"'{selected_program_title}' Jaise Aur Workout Programs:")

#     # --- Recommendation Logic ---
#     if selected_program_title not in df['title'].values:
#         st.error("Chuna gaya program nahi mil paaya.")
#     else:
#         idx = df[df['title'] == selected_program_title].index[0]
#         query_vector = tfidf_matrix[idx]
#         sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
#         similar_indices = sim_scores.argsort()[-top_n-1:-1][::-1]

#         recommendations = df.iloc[similar_indices].copy()
#         recommendations['similarity_score'] = sim_scores[similar_indices]

#         if recommendations.empty:
#             st.warning("Iske jaise aur programs nahi mil paaye.")
#         else:
#             num_cols = 3
#             cols = st.columns(num_cols)
#             for i, rec in enumerate(recommendations.itertuples()):
#                 with cols[i % num_cols]:
#                     with st.container(border=True):
#                         st.subheader(rec.title)
#                         st.markdown(f"**Similarity Score:** `{rec.similarity_score:.2f}`")
#                         with st.expander("Details Dekhein"):
#                             st.markdown(f"**Goal:** {getattr(rec, 'goal', 'N/A')}")
#                             st.markdown(f"**Level:** {getattr(rec, 'level', 'N/A')}")
#                             st.markdown(f"**Equipment:** {getattr(rec, 'equipment', 'N/A')}")
#                             st.write(getattr(rec, 'description', 'No description available.'))
# else:
#     st.info("Shuru karne ke liye, left sidebar se ek workout program chunein.")






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

st.success("App safaltapoorvak load ho gaya hai! Ab aap recommender ka istemal kar sakte hain.")

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




