import base64
import os
import requests
from dotenv import load_dotenv
import json

# load variables from .env
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

TOKEN_URL = "https://accounts.spotify.com/api/token"
BASE_URL = "https://api.spotify.com/v1"

def get_spotify_token():
    auth_string = CLIENT_ID + ":" + CLIENT_SECRET
    auth_bit = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bit), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": "Basic " + auth_base64,
               "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    result = requests.post(TOKEN_URL, headers=headers, data=data)

    json_data = json.loads(result.content)
    token = json_data["access_token"]

    return token

def get_auth_headers(token):
    headers = {"Authorization": "Bearer " + token}
    return headers

def search_artist(token, artist_name):
    url = f"{BASE_URL}/search"
    headers = get_auth_headers(token)
    query = f"q={artist_name}&type=artist&limit=2"
    query_url = url + "?" + query
    result = requests.get(query_url, headers=headers)
    json_data = json.loads(result.content)["artists"]["items"]
    if len(json_data) == 0:
        print("No artist with that name exists...")
        return None

    return json_data[0]

def get_songs_by_artist(token, artist_id):
    url = f"{BASE_URL}/artists/{artist_id}/top-tracks?country=US"
    auth_header = get_auth_headers(token)
    result = requests.get(url, headers=auth_header)
    json_data = json.loads(result.content)["tracks"]
    return json_data


def search_tracks(token, query, limit=10):
    url = f"{BASE_URL}/search"
    headers = get_auth_headers(token)
    params = {
        "q": query,
        "type": "track",
        "limit": limit,
    }
    result = requests.get(url, headers=headers, params=params)
    result.raise_for_status()
    return result.json()["tracks"]["items"]


def format_tracks(tracks):
    formatted = []
    for t in tracks:
        name = t["name"]
        artists = ", ".join(a["name"] for a in t["artists"])
        url = t["external_urls"]["spotify"]
        formatted.append(f"{name} — {artists} — {url}")
    return formatted


if __name__ == "__main__":
    token = get_spotify_token()
    artist = search_artist(token, "Bad Bunny")
    songs = get_songs_by_artist(token, artist["id"])
    for idk, song in enumerate(songs):
        print(f"{idk + 1}. {song['name']} - {song['external_urls']['spotify']}")
