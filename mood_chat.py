# mood_bot.py
import json
import requests

from spotify import get_spotify_token, search_tracks, format_tracks

MODEL = "llama3.2"
URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """
You are a planner for a music recommender.

The user will describe how they feel or what kind of music they want.
Your job is to output ONLY a JSON object with this structure:

{
  "mood": "<short mood description>",
  "query": "<a Spotify search query to find fitting songs>",
  "limit": <integer, 5-15>
}

Rules:
- Do add explanations
- Do NOT add markdown, or backticks.
- "query" should be something that works in the Spotify Search API, e.g.
  "sad indie rock", "chill edm", "upbeat latin reggaeton", etc.
- Keep "mood" short, like "sad but hopeful", "hype gym", "chill study".
- Use 10 as the default limit if the user doesn't say otherwise.
"""

def call_llm_for_plan(user_text: str) -> dict:
    """Ask Llama to turn user text into a JSON plan."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
    }

    resp = requests.post(URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    content = data["message"]["content"].strip()
    try:
        plan = json.loads(content)
    except json.JSONDecodeError:
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1:
            plan_str = content[first_brace:last_brace + 1]
            plan = json.loads(plan_str)
        else:
            raise RuntimeError(f"LLM response not valid JSON: {content}")

    return plan

def main():
    token = get_spotify_token()
    print("Spotify mood bot 🎧")
    print("Describe what you feel like listening to (or Ctrl+C to quit).")
    print("Example: 'sad girl indie', 'hype gym music', 'chill latin like Bad Bunny'.\n")

    while True:
        try:
            user_text = input("> ").strip()
            if not user_text:
                continue

            plan = call_llm_for_plan(user_text)
            mood = plan.get("mood", "unspecified mood")
            query = plan.get("query", user_text)
            limit = plan.get("limit", 10)

            print(f"\nInterpreted mood: {mood}")
            print(f"Searching Spotify for: \"{query}\" (limit {limit})\n")

            tracks = search_tracks(token, query, limit=limit)
            formatted = format_tracks(tracks)

            if not formatted:
                print("No tracks found, try describing it a bit differently.\n")
                continue

            for idx, line in enumerate(formatted, start=1):
                print(f"{idx}. {line}")
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()
