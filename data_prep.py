import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('model', exist_ok=True)

def load_and_clean_data():
    """Load raw anime data and clean it"""
    df = pd.read_csv('data/raw/anime.csv')
    
    print(f"Original dataset: {len(df)} entries")
    
    # Basic cleaning
    df = df.dropna(subset=['name'])  # Remove anime without names
    df['genre'] = df['genre'].fillna('Unknown')
    df['type'] = df['type'].fillna('TV')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(0)
    
    # Filter quality anime only
    df = df[
        (df['rating'] >= 6.0) &  # Good ratings only
        (df['episodes'] > 0) &   # Has episodes
        (df['episodes'] <= 1000) # No weird outliers
    ]
    
    print(f"After quality filter: {len(df)} entries")
    
    # Create composite profile for ML
    df['profile'] = (
        df['genre'].astype(str) + ' ' +
        df['type'].astype(str) + ' ' +
        df['name'].astype(str)
    )
    
    # Save cleaned dataset
    df.to_csv('data/processed/anime_cleaned.csv', index=False)
    print("✅ Cleaned dataset saved to data/processed/anime_cleaned.csv")
    
    return df

def prepare_tfidf_model(df):
    """Fit TF-IDF vectorizer and save it"""
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['profile'])
    
    # Save TF-IDF matrix and vectorizer
    pickle.dump(tfidf, open('model/tfidf_vectorizer.pkl', 'wb'))
    pickle.dump(tfidf_matrix, open('model/tfidf_matrix.pkl', 'wb'))
    pickle.dump(df, open('model/anime_data.pkl', 'wb'))
    
    print("✅ TF-IDF model prepared and saved")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    return tfidf, tfidf_matrix

if __name__ == "__main__":
    df = load_and_clean_data()
    tfidf, matrix = prepare_tfidf_model(df)
    print("🎉 Phase 1 Complete! Dataset ready for ML.")
