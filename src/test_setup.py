import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key and api_key != "your_api_key_here":
    print("ANTHROPIC_API_KEY is present and set.")
elif api_key == "your_api_key_here":
    print("ANTHROPIC_API_KEY found, but still set to the placeholder value.")
else:
    print("ANTHROPIC_API_KEY is missing.")
