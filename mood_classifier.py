# mood_classifier.py
import requests
import json

MODEL = "llama3.2"
URL = "http://localhost:11434/api/chat"

MOOD_LABELS = ["happy", "sad", "calm", "energetic", "focus", "party"]

def classify_mood(text: str):
    system_prompt = (
        "You are a mood classifier. "
        "Given the user's description of their mood or context, "
        "choose scores for these labels: "
        f"{', '.join(MOOD_LABELS)}.\n"
        "Respond ONLY with a JSON object of the form:\n"
        "{\n"
        '  \"labels\": [\"happy\", ...],\n'
        '  \"scores\": [0.9, ...]\n'
        "}\n"
        "Scores must be between 0 and 1."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    resp = requests.post(
        URL,
        json={"model": MODEL, "messages": messages, "stream": False},
    )
    resp.raise_for_status()
    data = resp.json()

    content = data["message"]["content"]

    # Try to parse JSON from the model output
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Super defensive fallback: if something goes wrong,
        # just guess a neutral label.
        return [("calm", 1.0)]

    labels = parsed.get("labels", [])
    scores = parsed.get("scores", [])

    # Zip labels + scores and sort descending by score
    pairs = list(zip(labels, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Filter only to the moods we know about, in case the model adds extras
    pairs = [(l, float(s)) for (l, s) in pairs if l in MOOD_LABELS]

    # Fallback if nothing valid
    if not pairs:
        pairs = [("calm", 1.0)]

    return pairs
