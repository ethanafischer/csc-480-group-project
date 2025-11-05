import sys
import json
import requests

MODEL = "llama3.2"
URL = "http://localhost:11434/api/chat"

def chat(chat_history):
    response = requests.post(
        URL,
        json={"model": MODEL, "messages": chat_history, "stream": True},
        stream=True,
    )
    reply_chunks = []

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        message = data.get("message")
        if message and "content" in message:
            piece = message["content"]
            sys.stdout.write(piece)
            sys.stdout.flush()
            reply_chunks.append(piece)
        if data.get("done"):
            break

    print()
    return "".join(reply_chunks)

def main():
    chat_history = []

    while True:
        try:
            user = input("> ").strip()
            if not user:
                continue
            chat_history.append({"role": "user", "content": user})
            assistant = chat(chat_history)
            chat_history.append({"role": "assistant", "content": assistant})
        except KeyboardInterrupt:
            print("\nBye!")
            break

if __name__ == "__main__":
    main()
