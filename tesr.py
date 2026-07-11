import requests
import json
import os
import time

# If you run this on the same Mac Mini, use localhost.
# If running from another computer on the network, use http://toggle.local:8000
BASE_URL = "http://localhost:8000"


def test_no_model_selection():
    """Tests the Gateway's ability to intercept requests without a model and fetch available ones."""
    print("\n--- TEST 1: Dynamic Model Selection ---")
    print("Sending chat request with NO model specified...")

    payload = {
        "messages": [{"role": "user", "content": "Hello! Are you there?"}]
    }

    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)

        if response.status_code == 400:
            data = response.json()
            print("✅ SUCCESS: Gateway intercepted the request.")
            print("Available Models from LM Studio:")
            for model in data.get("available_models", []):
                print(f"  - {model}")
            return data.get("available_models", [])
        else:
            print(f"❌ FAILED: Expected 400 Bad Request, got {response.status_code}")
            print(response.text)
            return []
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to Gateway. Is gateway.py running?")
        return []


def test_chat_completion(model_id):
    """Tests routing a properly formatted chat request through to LM Studio."""
    print(f"\n--- TEST 2: Chat Completion (Model: {model_id}) ---")
    print("Gateway should stop TTS (if running) and route this to LM Studio...")

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Write a one-sentence sci-fi story."}
        ],
        "temperature": 0.7
    }

    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS: LLM Response:")
        print(data["choices"][0]["message"]["content"])
    else:
        print(f"❌ FAILED: Status {response.status_code}")
        print(response.text)


def test_tts_generation():
    """Tests the gateway unloading LM Studio and spinning up the Kokoro Docker container."""
    print("\n--- TEST 3: Text-to-Speech Generation ---")
    print("Gateway should unload the LLM from memory and spin up the Docker container...")

    payload = {
        "input": "This is a test of the automatic memory orchestration system.",
        "voice": "af_bella"
    }

    # Note: This will likely take a few seconds as the container spins up
    response = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)

    if response.status_code == 200:
        print("✅ SUCCESS: Audio generation triggered and completed!")
        print("Gateway Response:", response.json())
    else:
        print(f"❌ FAILED: Status {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("Initializing Gateway Tests...")

    # Ensure LM Studio is open and running its local server on port 1234
    print("Make sure LM Studio local server is running, and gateway.py is running in another terminal.\n")
    time.sleep(2)

    # 1. Ask the Gateway what models are available
    available_models = test_no_model_selection()

    # 2. If we found models, pick the first one and send a real chat request
    if available_models:
        selected_model = available_models[0]
        test_chat_completion(selected_model)
    else:
        print("Skipping Test 2: No models available to test with. Did you download one in LM Studio?")

    # 3. Test the TTS Docker swap
    test_tts_generation()

    print("\nAll tests complete! Check your Gateway terminal to see the orchestration logs.")