#!/usr/bin/env python3
"""
Simple interactive prompt loop for Atlas AI Agent
Talk to the router at http://127.0.0.1:8083
"""

import sys
import json
import requests
from typing import Optional

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

ROUTER_URL = "http://127.0.0.1:8083/v1/chat/completions"
DEFAULT_MODEL = "general_fast"
DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOKENS = 1024


def ask_atlas(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system: str = "You are Atlas, a helpful local AI agent using multiple models via router."
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(ROUTER_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        return f"Error communicating with router: {e}\n(Response: {resp.text if 'resp' in locals() else 'no response'})"
    except (KeyError, json.JSONDecodeError):
        return f"Invalid response format from router.\nRaw: {resp.text if 'resp' in locals() else 'no response'}"


def main():
    if len(sys.argv) > 1:
        # One-shot mode: python prompt.py "your question"
        query = " ".join(sys.argv[1:])
        print("Atlas:", ask_atlas(query))
        return

    # Interactive mode
    print("Atlas interactive mode (Ctrl+C or empty line + Enter to exit)")
    print("─" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            print("\nAtlas: ", end="", flush=True)
            response = ask_atlas(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            continue


if __name__ == "__main__":
    main()