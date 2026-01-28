from flask import Flask, render_template, request
from model.recommender import AnimeRecommender
import requests
import os
import re

app = Flask(__name__)
rec = AnimeRecommender()

def get_anime_poster(title):
    """Fetch real poster from Jikan API"""
    try:
        # Clean title for search
        search_title = re.sub(r'[^\w\s]', ' ', title).strip()
        
        # Search Jikan API
        url = f"https://api.jikan.moe/v4/anime?q={search_title}&limit=1"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                return {
                    'image_url': data['data'][0]['images']['jpg']['large_image_url'],
                    'mal_url': data['data'][0]['url'],
                    'rating': data['data'][0].get('myanimelist_score', 'N/A'),
                    'episodes': data['data'][0].get('episodes', 'N/A')
                }
    except:
        pass
    
    # Fallback poster
    return {
        'image_url': f"https://via.placeholder.com/240x340/1a1a1a/888888888?text={title[:12]}...",
        'mal_url': '#',
        'rating': 'N/A',
        'episodes': 'N/A'
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get form data
        genre = request.form.get('genre', 'Action')
        mood = request.form.get('mood', '')
        style = request.form.get('style', '')
        reference_title = request.form.get('reference_title', '').strip()
        
        # Get ML recommendations
        results = rec.get_recommendations(genre, mood, style, reference_title)
        
        # Fetch real posters for main + backups
        if results['main']:
            results['main']['poster'] = get_anime_poster(results['main']['title'])
        for backup in results['backups']:
            backup['poster'] = get_anime_poster(backup['title'])
        
        return render_template('result.html', 
                             main=results['main'],
                             backups=results['backups'],
                             query_used=results['query_used'])
        
    except Exception as e:
        return render_template('result.html', 
                             error=f"Oops! Try different inputs: {str(e)}",
                             main=None,
                             backups=[])

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("🚀 Anirex starting at http://localhost:5000")
    print("✅ ML + Jikan API ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)
