# recommender.py
import numpy as np

def filter_tracks_for_mood(df, mood: str, n: int = 15):
    q = df

    # Rule-based constraints per mood label
    if mood == "happy":
        q = q[(q["valence"] > 0.7) & (q["energy"] > 0.5)]

    elif mood == "sad":
        # lower valence, slower tempo
        if "tempo" in q.columns:
            q = q[(q["valence"] < 0.4) & (q["tempo"] < 115)]
        else:
            q = q[(q["valence"] < 0.4)]

    elif mood == "calm":
        if "acousticness" in q.columns:
            q = q[(q["energy"] < 0.5) & (q["acousticness"] > 0.5)]
        else:
            q = q[(q["energy"] < 0.5)]

    elif mood == "energetic":
        q = q[(q["energy"] > 0.75) & (q["danceability"] > 0.6)]

    elif mood == "focus":
        speechiness_col = "speechiness" if "speechiness" in q.columns else None
        q = q[
            (q["energy"].between(0.3, 0.7)) &
            (q["valence"].between(0.3, 0.7))
        ]
        if speechiness_col:
            q = q[q[speechiness_col] < 0.33]

    elif mood == "party":
        q = q[(q["danceability"] > 0.7) & (q["energy"] > 0.7)]

    # you can add more moods here with custom constraints

    if len(q) == 0:
        return []

    # sample N tracks to show variety
    sample = q.sample(min(n, len(q)), random_state=np.random.randint(0, 1_000_000))

    results = []
    for _, row in sample.iterrows():
        results.append(
            {
                "track_name": row["track_name"],
                "artists": row["artists"],
                "genre": row["track_genre"] if "track_genre" in row else None,
            }
        )

    return results
