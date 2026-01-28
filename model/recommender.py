import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

class AnimeRecommender:
    def __init__(self):
        """Load pre-trained TF-IDF model and anime data"""
        self.anime_df = pickle.load(open('model/anime_data.pkl', 'rb'))
        self.tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))
        self.tfidf_matrix = pickle.load(open('model/tfidf_matrix.pkl', 'rb'))
        self.anime_indices = {title: idx for idx, title in enumerate(self.anime_df['name'])}
        
    def preprocess_user_input(self, genre, mood, style, reference_title):
        """Convert user selections into ML query"""
        mood_keywords = {
            'Dark': 'Psychological Horror Dark Thriller',
            'Wholesome': 'Slice of Life Comedy Heartwarming',
            'Motivational': 'Sports Shounen Action Inspirational',
            'Chill': 'Slice of Life Music Iyashikei',
            'Emotional': 'Drama Romance Tragedy'
        }
        
        style_keywords = {
            'Classic': 'Classic 90s 80s',
            'Modern': 'New 2020s 2010s',
        }
        
        query_parts = [genre]
        if mood and mood in mood_keywords:
            query_parts.append(mood_keywords[mood])
        if style and style in style_keywords:
            query_parts.append(style_keywords[style])
            
        query_text = ' '.join(query_parts)
        
        # Handle reference anime
        if reference_title and str(reference_title).strip():
            ref_anime = self.find_similar_anime_title(reference_title)
            if ref_anime is not None:
                query_text = f"{ref_anime['name']} {query_text}"

        return query_text
    
    def find_similar_anime_title(self, user_title):
        """Find closest matching anime title using fuzzy matching"""
        user_title = str(user_title).lower().strip()
        
        # Exact match first
        for title, idx in self.anime_indices.items():
            if title.lower() == user_title:
                return self.anime_df.iloc[idx]
        
        # Fuzzy match
        all_titles = list(self.anime_indices.keys())
        matches = get_close_matches(user_title, [t.lower() for t in all_titles], n=1, cutoff=0.6)
        
        if matches:
            matched_title = next((t for t in all_titles if t.lower() == matches[0]), None)
            if matched_title:
                idx = self.anime_indices[matched_title]
                return self.anime_df.iloc[idx]
        
        return None
    
    def get_recommendations(self, genre, mood, style, reference_title, top_k=4):
        """Main recommendation function"""
        query_text = self.preprocess_user_input(genre, mood, style, reference_title)
        query_vec = self.tfidf.transform([query_text])
        
        # Calculate base similarity scores
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Style-aware adjustment at the score level so that
        # Classic vs Modern can actually change the ranking
        adjusted_scores = sim_scores.copy()
        if style in ('Classic', 'Modern') and 'members' in self.anime_df.columns:
            members = self.anime_df['members'].fillna(0).astype(float).to_numpy()
            style_factors = np.ones_like(adjusted_scores, dtype=float)
            if style == 'Classic':
                # Downweight very popular titles to surface less mainstream / older picks
                style_factors = 1.0 / (1.0 + (members / 1_000_000.0))
            elif style == 'Modern':
                # Slightly boost popular titles as a proxy for modern hits
                style_factors = 1.0 + (members / 2_000_000.0)
            adjusted_scores = sim_scores * style_factors

        # Rank by adjusted scores. We'll scan a larger window so that:
        # - genre filtering doesn't accidentally return off-genre anime
        # - same-series filtering doesn't wipe out all candidates for reference-based queries
        ranked_indices = np.argsort(adjusted_scores)[::-1]
        window_size = top_k * (50 if (reference_title and str(reference_title).strip()) else 20)
        window_size = max(window_size, 200)
        candidate_indices = ranked_indices[:window_size]
        
        # Determine reference series keys for same-series filtering
        reference_series_keys = []
        ref_anime = None
        if reference_title and str(reference_title).strip():
            raw_key = str(reference_title).lower().strip()
            if raw_key:
                reference_series_keys.append(raw_key)
            # Also use the resolved anime title, if we can find one
            ref_anime = self.find_similar_anime_title(reference_title)
            if ref_anime is not None and 'name' in ref_anime:
                resolved_key = str(ref_anime['name']).lower().strip()
                if resolved_key and resolved_key not in reference_series_keys:
                    reference_series_keys.append(resolved_key)

        recommendations = []
        requested_genre = str(genre).lower().strip() if genre else ""

        for idx in candidate_indices:
            if adjusted_scores[idx] > 0.1:  # Similarity threshold based on adjusted similarity
                anime = self.anime_df.iloc[idx]
                
                # FIX: Extract scalar values properly
                title = str(anime['name'])
                anime_genre = str(anime['genre'])
                anime_type = str(anime['type'])
                episodes = float(anime['episodes']) if pd.notna(anime['episodes']) else 0
                rating = float(anime['rating']) if pd.notna(anime['rating']) else 0
                similarity = float(adjusted_scores[idx])

                # Enforce selected genre (avoid off-genre recommendations like "Comedy" -> Thriller)
                if requested_genre:
                    anime_genres = [g.strip().lower() for g in str(anime_genre).split(",") if g.strip()]
                    if requested_genre not in anime_genres:
                        continue

                # Same-series detection:
                # 1) Skip exact same anime as the reference (by anime_id when available)
                # 2) Skip any title that contains either the raw user reference or the resolved series name
                same_series = False
                title_lower = title.lower()

                # 1) Exact anime match
                if ref_anime is not None and 'anime_id' in self.anime_df.columns and 'anime_id' in ref_anime:
                    try:
                        if int(anime['anime_id']) == int(ref_anime['anime_id']):
                            same_series = True
                    except Exception:
                        pass

                # 2) Name-based same-series detection
                if not same_series and reference_series_keys:
                    same_series = any(key in title_lower for key in reference_series_keys if key)

                if same_series:
                    continue

                recommendations.append({
                    'title': title,
                    'genre': anime_genre,
                    'type': anime_type,
                    'episodes': int(episodes),
                    'rating': round(rating, 2),
                    'similarity': round(similarity, 3),
                    'reason': self.generate_explanation(anime, query_text)
                })

                # We only need enough to fill main + backups
                if len(recommendations) >= top_k:
                    break
        
        # Return structured results
        main = recommendations[0] if recommendations else None
        backups = recommendations[1:4] if len(recommendations) > 1 else []
        
        return {
            'main': main,
            'backups': backups,
            'query_used': query_text,
            'total_matches': len(recommendations)
        }
    
    def generate_explanation(self, anime, query_text):
        """Generate human-readable explanation"""
        try:
            genres = str(anime['genre']).split(',')[:3]
            genres = [g.strip() for g in genres if g.strip()]
            genre_text = ', '.join(genres) if genres else 'Various'
        except:
            genre_text = 'Various'
            
        try:
            anime_type = str(anime['type'])
        except:
            anime_type = 'TV'
            
        return f"{anime_type} format. Genres: {genre_text}"
