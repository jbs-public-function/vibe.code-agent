import json
import sys
import urllib.request

from openai import OpenAI

# Define local ports mapped in your Makefile
MODEL_ROSTER = {
    "8010": "DeepSeek-R1 (14B)",
    "8020": "Llama-3.1 (8B)",
    "8030": "Qwen-Coder (7B)",
}


def get_active_client():
    """Scans local ports and dynamically requests the active model ID from vLLM."""
    import socket

    # Loop through ports to see which one is actively serving vLLM data
    for port, friendly_name in MODEL_ROSTER.items():
        # First check if the port is physically open
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", int(port))) != 0:
                continue  # Port is closed, skip to next one

        # Port is open! Let's verify it's actually responding with vLLM model data
        try:
            url = f"http://localhost:{port}/v1/models"
            with urllib.request.urlopen(url, timeout=1) as response:
                data = json.loads(response.read().decode())
                exact_model_id = data["data"][0][
                    "id"
                ]  # Fixed: read first index of the models data array

            print(f"Connected to: [ {friendly_name} ] on port {port}")
            print(f"Active Model ID: {exact_model_id}\n")

            return OpenAI(
                base_url=f"http://localhost:{port}/v1", api_key="local"
            ), exact_model_id
        except Exception:
            # Port was open but didn't return valid vLLM data, look at next port
            continue

    print("ERROR: No active vLLM containers found. Run 'make <model>' first.")
    sys.exit(1)


# Initialize the active connection dynamically
client, active_model_repo = get_active_client()

# Your testing prompt
user_prompt = "Write a bullet-point resume entry describing setting up high-performance local LLM inference engines using vLLM and Docker containers on Linux."

# Stream the tokens in real time
response = client.chat.completions.create(
    model=active_model_repo,
    messages=[{"role": "user", "content": user_prompt}],
    temperature=0.7,
    stream=True,
)

for chunk in response:
    # Ensure choices exist and the list isn't empty before reading the delta
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
