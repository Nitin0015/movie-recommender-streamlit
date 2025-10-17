import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
import pandas as pd
import requests
from typing import Dict, List

# --- CONFIG ---
API_KEY = "a2e8cd5f3dd3329f5ded63fbfe3625e9"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500/"
COUNTRY_CODE = "US"


# --- API HELPERS ---
def fetch_movie_details(movie_id: int) -> Dict:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": "en-US"}
    try:
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"error": "Failed to fetch movie data."}
    genres = ', '.join(g['name'] for g in data.get('genres', []))
    poster_url = TMDB_IMG_BASE + data.get('poster_path', "") if data.get('poster_path') else ""
    tmdb_link = f"https://www.themoviedb.org/movie/{movie_id}"
    homepage = data.get('homepage') or tmdb_link
    year = (data.get('release_date') or "")[:4]
    return {
        "title": data.get('title', ""),
        "poster_url": poster_url,
        "genres": genres,
        "overview": data.get('overview', "No description available."),
        "homepage": homepage,
        "year": year or "Unknown",
        "tmdb_link": tmdb_link
    }

def fetch_providers(movie_id: int) -> str:
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers"
    params = {"api_key": API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json().get('results', {}).get(COUNTRY_CODE, {})
    except Exception:
        return "No info"
    providers = []
    logos = []
    for k in ['flatrate', 'rent', 'buy']:
        for p in data.get(k, []):
            name = p.get('provider_name')
            logo = p.get('logo_path')
            logo_html = f"<img src='https://image.tmdb.org/t/p/w45/{logo}' style='vertical-align:middle;margin-right:3px;' alt='{name}'>" if logo else ""
            providers.append(f"{logo_html} <b>{name}</b> ({k.capitalize()})")
    return "<br>".join(providers) if providers else "No streaming info"

def recommend(movie_title: str, movies: pd.DataFrame, similarity: np.ndarray) -> List[Dict]:
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
    except Exception:
        return [{"error": "Movie not found in DB"}]
    distances = similarity[movie_index]
    chosen = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:8]
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_movie_details, movies.iloc[i[0]].movie_id)
            for i in chosen
        ]
        details = [f.result() for f in futures]
        for idx, info in enumerate(details):
            providers_html = fetch_providers(movies.iloc[chosen[idx][0]].movie_id)
            info["providers"] = providers_html
            results.append(info)
    return results

@st.cache_resource
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    movies_df = pd.DataFrame(movies_dict)
    return movies_df, similarity

# --- UI ---
def render_profile_card():
    st.markdown("""
        <div style="position:fixed;top:32px;right:20px;background:#151a2d;color:#fff;padding:14px 20px;z-index:100;border-radius:10px;border:1px solid #7bc9ff;box-shadow:0 0 10px 0 #b5eaff;">
        <b>Nitin Chowdary</b><br>
        Data Science Graduate<br>
        Movie Recommender SDE6 Demo 🚀
        </div>
        """, unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Movie Recommender SDE6", layout="wide", initial_sidebar_state="expanded")
    st.markdown("<h1 style='text-align: center; color: #00aced;font-size:3em;'>🍿 Movie Recommender System</h1>", unsafe_allow_html=True)
    render_profile_card()
    st.markdown("""
        <style>
        .stApp {background-image: linear-gradient(120deg, #313549, #215eaa 90%);}
        .stButton > button {background: #00aced;border-radius:9px;color:#fff;}
        .stSelectbox label {font-size:1.2em;font-weight:bold;}
        </style>""", unsafe_allow_html=True)

    movies, similarity = load_data()

    st.markdown("<p style='text-align: center; font-size: 1.4em;'>Find your next favorite movie!</p>", unsafe_allow_html=True)

    search_q = st.text_input("🔍 Start typing any movie to filter...", "")
    if search_q:
        filtered_titles = movies[movies['title'].str.contains(search_q, case=False, regex=False)]['title'].values
    else:
        filtered_titles = movies['title'].values

    selected_movie = st.selectbox('Choose a movie for recommendations:', filtered_titles, index=0)

    if st.button("🚀 Recommend", help="Get the best recommendations!"):
        recs = recommend(selected_movie, movies, similarity)
        st.markdown("---")
        st.subheader(f"Recommendations Based on {selected_movie}:")
        cols = st.columns(4)
        for idx, col in enumerate(cols):
            if idx >= len(recs):
                continue
            info = recs[idx]
            if info.get("error"):
                col.error(info["error"])
                continue
            col.markdown(f"<div style='text-align:center;'><img src='{info['poster_url']}' width='175'/></div>", unsafe_allow_html=True)
            col.markdown(f"<b>{info['title']} ({info['year']})</b>", unsafe_allow_html=True)
            genre_str = f"Genres: <i>{info['genres']}</i>"
            providers_str = info['providers']
            with col.expander(f"Details", expanded=False):
                st.markdown(f"<b>{info['title']}</b> <span style='font-size:1.1em;'>({info['year']})</span>", unsafe_allow_html=True)
                st.image(info["poster_url"], caption=info["title"], width=350)
                st.write(genre_str, unsafe_allow_html=True)
                st.write(info["overview"])
                st.markdown(f"<b>Where to Watch:</b><br>{providers_str}", unsafe_allow_html=True)
                st.markdown(f"<a href='{info['homepage']}' target='_blank'>TMDB Homepage</a> | <a href='{info['tmdb_link']}' target='_blank'>See on TMDB</a>", unsafe_allow_html=True)
        st.markdown("---")
    st.caption("Created with ❤️ by Nitin Chowdary")

main()
