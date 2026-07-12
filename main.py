import os
import time
import subprocess
import httpx
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import docker
except Exception as exc:  # pragma: no cover - depends on local environment
    docker = None
    docker_import_error = exc
else:
    docker_import_error = None

docker_not_found_error = getattr(getattr(docker, "errors", None), "NotFound", Exception)

app = FastAPI(title="Gate8 Resource Gateway")

# Enable CORS so web interfaces can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

docker_client = None
if docker is not None:
    try:
        docker_client = docker.from_env()
    except Exception as exc:
        docker_import_error = exc

# Configuration of the API
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
TTS_Container_name = "kokoro-tts-server"
TTS_Container_image = "ghcr.io/remsky/kokoro-fastapi-cpu:latest"


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7


class SpeechRequest(BaseModel):
    input: str
    voice: str = "af_bella"  # Default voice
    speed: float = 1.0  # Default speed


def _docker_unavailable_detail() -> str:
    if docker_import_error is None:
        return "Docker is unavailable: No Docker SDK found and 'docker' CLI command failed."

    error_str = str(docker_import_error)
    detail = f"Docker is unavailable: {error_str}"

    if "distutils" in error_str.lower():
        detail += " (Note: Python 3.12+ requires 'pip install setuptools' for the Docker SDK to work)."

    return detail


def _run_docker_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Helper to run raw Docker CLI commands if the SDK fails."""
    # Try common paths for Docker, especially useful on macOS or if PATH is restricted
    paths_to_try = ["docker", "/usr/local/bin/docker", "/usr/bin/docker", "/opt/homebrew/bin/docker"]

    last_error = None
    for path in paths_to_try:
        try:
            return subprocess.run([path, *args], capture_output=True, text=True)
        except FileNotFoundError as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    raise FileNotFoundError("docker command not found")


def _wait_for_tts_ready(max_attempts=30, delay=1.0):
    """Poll Kokoro until it responds, with timeout."""
    for attempt in range(max_attempts):
        try:
            # Use blocking httpx to check if service is up
            with httpx.Client(timeout=2.0) as client:
                resp = client.get("http://localhost:8880/docs")
                if resp.status_code < 500:
                    print(f"[Orchestrator] TTS container ready (attempt {attempt + 1})")
                    return
        except Exception as e:
            pass

        if attempt < max_attempts - 1:
            print(f"[Orchestrator] Waiting for TTS... ({attempt + 1}/{max_attempts})")
            time.sleep(delay)

    raise HTTPException(status_code=503, detail="TTS container failed to become ready after 30 seconds")


def _ensure_tts_container_running():
    """Start or unpause the TTS container using the Python SDK first, then fall back to the Docker CLI."""
    if docker_client is not None and docker is not None:
        try:
            container = docker_client.containers.get(TTS_Container_name)
            if container.status == "paused":
                print(f"[Orchestrator] Unpausing TTS container {TTS_Container_name}...")
                container.unpause()
            elif container.status != "running":
                print(f"[Orchestrator] Starting TTS container {TTS_Container_name}...")
                container.start()
            return
        except docker_not_found_error:
            docker_client.containers.run(
                TTS_Container_image,
                name=TTS_Container_name,
                detach=True,
                ports={'8880/tcp': 8880},
                mem_limit="2g",
            )
            return
        except Exception as exc:
            print(f"[Orchestrator] Docker SDK failed, falling back to Docker CLI: {exc}")

    try:
        if docker_client is None:
            print("[Orchestrator] Docker SDK not available. Using Docker CLI...")
        inspect = _run_docker_cli(["inspect", "-f", "{{.State.Status}}", TTS_Container_name])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_docker_unavailable_detail()) from exc

    if inspect.returncode == 0:
        status_str = inspect.stdout.strip().lower()
        if status_str == "paused":
            print(f"[Orchestrator] Unpausing TTS container {TTS_Container_name} via CLI...")
            _run_docker_cli(["unpause", TTS_Container_name])
        elif status_str != "running":
            print(f"[Orchestrator] Starting TTS container {TTS_Container_name} via CLI...")
            start = _run_docker_cli(["start", TTS_Container_name])
            if start.returncode != 0:
                raise HTTPException(status_code=503,
                                    detail=f"Failed to start TTS container: {start.stderr.strip() or start.stdout.strip()}")
        return

    run = _run_docker_cli([
        "run",
        "-d",
        "--name",
        TTS_Container_name,
        "-p",
        "8880:8880",
        "--memory=2g",
        TTS_Container_image,
    ])
    if run.returncode != 0:
        raise HTTPException(status_code=503,
                            detail=f"Failed to create TTS container: {run.stderr.strip() or run.stdout.strip()}")


def _pause_tts_container_cli():
    """Pauses the container via the terminal if the Python library hangs."""
    try:
        inspect = _run_docker_cli(["inspect", "-f", "{{.State.Status}}", TTS_Container_name])
    except FileNotFoundError:
        return

    if inspect.returncode == 0 and inspect.stdout.strip().lower() == "running":
        print(f"[Orchestrator] Pausing TTS container {TTS_Container_name} via CLI...")
        _run_docker_cli(["pause", TTS_Container_name])


def safely_pause_tts():
    """Checks if the TTS Docker container is running and pauses it to free CPU/RAM."""
    print("[Orchestrator] Ensuring TTS container is paused...")
    if docker_client is None or docker is None:
        print("[Orchestrator] Docker SDK not available. Using Docker CLI for pausing...")
        _pause_tts_container_cli()
        return

    try:
        container = docker_client.containers.get(TTS_Container_name)
        if container.status == "running":
            print(f"[Orchestrator] Pausing TTS container {TTS_Container_name} via SDK...")
            container.pause()
            time.sleep(0.5)
    except docker_not_found_error:
        pass
    except Exception as e:
        print(f"[Orchestrator] Warning: Docker SDK failed to pause TTS: {e}")
        _pause_tts_container_cli()


async def safely_unload_llm():
    """Tells LM Studio to eject loaded models from memory."""
    api_v1_url = LM_STUDIO_URL.replace("/v1", "/api/v1")
    api_v0_url = LM_STUDIO_URL.replace("/v1", "/api/v0")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            loaded_instances = []

            # 1. Try /api/v1/models (preferred)
            v1_resp = await client.get(f"{api_v1_url}/models")
            if v1_resp.status_code == 200:
                v1_data = v1_resp.json()
                models_list = v1_data.get("models", [])
                for m in models_list:
                    for instance in m.get("loaded_instances", []):
                        if instance.get("id"):
                            loaded_instances.append(instance["id"])

            # 2. Try /api/v0/models if v1 didn't yield anything or failed
            if not loaded_instances:
                v0_resp = await client.get(f"{api_v0_url}/models")
                if v0_resp.status_code == 200:
                    v0_data = v0_resp.json()
                    models_list = v0_data.get("data", [])
                    loaded_instances = [m["id"] for m in models_list if m.get("state") == "loaded"]

            # 3. Final fallback: standard OpenAI /v1/models
            if not loaded_instances:
                fallback_resp = await client.get(f"{LM_STUDIO_URL}/models")
                if fallback_resp.status_code == 200:
                    models_data = fallback_resp.json().get("data", [])
                    loaded_instances = [m["id"] for m in models_data]

            if not loaded_instances:
                print("[Orchestrator] No LLMs are currently loaded.")
                return

            # Unload each instance safely
            for instance_id in loaded_instances:
                print(f"[Orchestrator] Ejecting LLM instance: {instance_id}")

                unload_resp = await client.post(f"{api_v1_url}/models/unload", json={"instance_id": instance_id})
                if unload_resp.status_code != 200:
                    unload_resp = await client.post(f"{api_v1_url}/models/unload", json={"model": instance_id})
                if unload_resp.status_code != 200:
                    unload_resp = await client.post(f"{api_v0_url}/models/unload", json={"model": instance_id})

                if unload_resp.status_code == 200:
                    print(f"[Orchestrator] Successfully unloaded {instance_id}")
                else:
                    print(f"[Orchestrator] Failed to unload {instance_id}: {unload_resp.text}")

        except Exception as e:
            print(f"[Orchestrator] Error during LLM unloading: {e}")


async def get_lm_studio_models() -> List[str]:
    """Fetches the list of available models directly from LM Studio."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{LM_STUDIO_URL}/models")
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            return []
        except Exception:
            return []


# ---------------------------------------------------------
# Serve the Frontend Web UIs
# ---------------------------------------------------------
@app.get("/")
async def serve_chat_assistant():
    """Serves the index.html file (Chat Assistant)."""
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "index.html not found."})


@app.get("/reader")
async def serve_novel_reader():
    """Serves the WebNovelReader.html file."""
    file_path = os.path.join(os.path.dirname(__file__), "WebNovelReader.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "WebNovelReader.html not found."})


@app.post("/v1/chat/completions")
async def handle_chat(payload: ChatRequest):
    # Enforce the 8GB RAM limit by pausing TTS first
    safely_pause_tts()

    # Handle missing or "default" model selection
    if not payload.model or payload.model.lower() == "default":
        available_models = await get_lm_studio_models()
        if not available_models:
            raise HTTPException(status_code=503, detail="No models are currently available or downloaded in LM Studio.")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "No model specified.",
                "message": "Please select a model and include it in your request payload.",
                "available_models": available_models
            }
        )

    # Forward the request to LM Studio
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print(f"[Orchestrator] Routing chat request to LM Studio (Model: {payload.model})...")
            response = await client.post(
                f"{LM_STUDIO_URL}/chat/completions",
                json=payload.model_dump()
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())

            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to LM Studio: {str(e)}")


@app.post("/v1/audio/speech")
async def handle_speech(payload: SpeechRequest):
    # Enforce the 8GB RAM limit by unloading the LLM first
    await safely_unload_llm()

    try:
        print("[Orchestrator] Spinning up TTS Container...")
        _ensure_tts_container_running()
        print("[Orchestrator] Waiting for TTS container to become ready...")
        _wait_for_tts_ready()

        print("[Orchestrator] Routing speech request to Kokoro...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                tts_response = await client.post(
                    "http://localhost:8880/v1/audio/speech",
                    json=payload.model_dump()
                )

                if tts_response.status_code != 200:
                    raise HTTPException(status_code=tts_response.status_code,
                                        detail=tts_response.text or "TTS Generation Failed")

                audio_content = tts_response.content

            except httpx.RequestError as e:
                raise HTTPException(status_code=503, detail=f"Cannot connect to TTS container: {str(e)}")

        print("[Orchestrator] Audio generated. Pausing TTS container...")
        safely_pause_tts()

        # Return the raw audio bytes as a streamable response
        from fastapi.responses import Response
        return Response(content=audio_content, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS Pipeline Error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)