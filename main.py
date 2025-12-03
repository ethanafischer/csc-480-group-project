# main.py
from mood_classifier import classify_mood
from data_loader import load_tracks
from recommender import filter_tracks_for_mood

def main():
    df = load_tracks("dataset.csv")

    print("Spotify mood recommender")
    print("Describe what you want (e.g. 'sad rock like Paramore', 'chill study music',")
    print("'something like Bad Bunny but more mellow').\n")

    while True:
        text = input("Describe your mood or situation (or 'quit'):\n> ").strip()
        if not text:
            continue
        if text.lower() in {"q", "quit", "exit"}:
            break

        label_scores = classify_mood(text)

        print("\nPredicted moods:")
        for label, score in label_scores:
            print(f"- {label} ({score:.2f})")

        top_label = label_scores[0][0]

        recs = filter_tracks_for_mood(df, top_label, n=10)

        print(f"\nRecommended tracks for mood '{top_label}':\n")
        if not recs:
            print("No tracks matched the criteria. This is an edge case you can mention in the report.")
        else:
            for i, t in enumerate(recs, start=1):
                genre = f" [{t['genre']}]" if t.get("genre") else ""
                print(f"{i}. {t['track_name']} – {t['artists']}{genre}")
        print()

if __name__ == "__main__":
    main()
