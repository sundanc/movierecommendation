import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    file_path = 'movie_dataset.csv'  # Adjust this if needed
    return pd.read_csv(file_path)

def preprocess_movies_data(movies_df):
    # Fill missing values
    for column in ['genres', 'keywords', 'overview', 'cast', 'director', 'tagline']:
        movies_df[column] = movies_df[column].fillna('')
    
    # Combine features to create a "combined_features" column
    movies_df['combined_features'] = (
        movies_df['genres'] + ' ' +
        movies_df['keywords'] + ' ' +
        movies_df['overview'] + ' ' +
        movies_df['cast'] + ' ' +
        movies_df['director'] + ' ' +
        movies_df['tagline']
    )
    
    return movies_df

def calculate_similarity(movies_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies_df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_index_from_title(movies_df, title):
    title = title.lower().strip()
    movies_df['normalized_title'] = movies_df['title'].str.lower().str.strip()
    if title in movies_df['normalized_title'].values:
        return movies_df[movies_df['normalized_title'] == title].index[0]
    else:
        return None

def get_title_from_index(movies_df, index):
    return movies_df.iloc[index]['title']

def recommend_movies(movies_df, input_title, cosine_sim):
    movie_idx = get_index_from_title(movies_df, input_title)
    
    if movie_idx is None:
        return None
    
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    if len(sorted_sim_scores) < 2:
        return None
    
    recommendations = []
    for i in sorted_sim_scores[1:6]:
        recommended_idx = i[0]
        recommendations.append(get_title_from_index(movies_df, recommended_idx))
    
    return recommendations

def main():
    st.title("Movie Recommendation System")
    
    movies_df = load_data()
    
    movies_df = preprocess_movies_data(movies_df)
    
    cosine_sim = calculate_similarity(movies_df)

    input_movie = st.text_input("Enter a Movie Title:", "")
    
    if input_movie:
        movie_idx = get_index_from_title(movies_df, input_movie)
        
        if movie_idx is not None:
            st.subheader(f"Details for '{input_movie}':")
            movie_details = movies_df.iloc[movie_idx][['title', 'genres', 'overview']]
            st.write(movie_details)
            
            st.subheader(f"Recommended Movies Similar to '{input_movie}':")
            recommendations = recommend_movies(movies_df, input_movie, cosine_sim)
            
            if recommendations:
                st.write(", ".join(recommendations))
            else:
                st.write("No similar movies found.")
        else:
            st.write(f"Movie '{input_movie}' not found in the dataset.")
                
if __name__ == "__main__":
    main()
