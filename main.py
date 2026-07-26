import asyncio
import os
import time
import asyncio
import hashlib
import subprocess
import tempfile
import httpx
from typing import List
from exporter import exporter
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles


# Import R2 Storage Helpers
from r2_storage import (
    upload_audio_to_r2,
    generate_presigned_download_url,
    get_r2_storage_list,
    download_file_from_r2
)


try:
    import docker
except Exception as exc:
    docker = None
    docker_import_error = exc
else:
    docker_import_error = None

docker_not_found_error = getattr(getattr(docker, "errors", None), "NotFound", Exception)

app = FastAPI(title="Gate8 Resource Gateway")

app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/js", StaticFiles(directory="js"), name="js")


# Enable CORS for frontend and edge worker interactions
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

# --- Service Configurations ---
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
TTS_Container_name = "kokoro-tts-server"
TTS_Container_image = "ghcr.io/remsky/kokoro-fastapi-cpu:latest"

METADEFENDER_API_KEY = os.getenv("METADEFENDER_API_KEY", "")
METADEFENDER_URL = "https://api.metadefender.com/v4"

RESPONSE_FORMAT_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

# In-memory background task tracking
audiobook_jobs = {}

import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger("gate8.resource_manager")


class SmartResourceManager:
    def __init__(self, total_ram_mb: int = 8192):
        self.llm_active: bool = False
        self.tts_active: bool = False
        self.active_llm_model: str | None = None
        self.lock = asyncio.Lock()

        # Approximate memory footprint baselines
        self.vram_total_mb: int = total_ram_mb
        self.llm_vram_mb: int = 2200  # Baseline ~2.2GB for 3B quantized LLM
        self.tts_vram_mb: int = 1400  # Baseline ~1.4GB for Kokoro Docker Pod
        self.base_os_mb: int = 3500  # Reserved for macOS base system

    async def prepare_for_tts(self) -> Dict[str, Any]:
        """
        Unloads active LLMs from LM Studio VRAM and ensures Kokoro TTS is running
        to avoid system swap thrashing on 8GB Unified Memory setups.
        """
        async with self.lock:
            logger.info("[SmartResourceManager] Prepping memory for Kokoro TTS...")

            # 1. Unload LLM from VRAM
            try:
                await safely_unload_llm()
                self.llm_active = False
            except Exception as e:
                logger.warning(f"[SmartResourceManager] LLM unload warning: {e}")

            # 2. Spin up/unpause Kokoro TTS container
            try:
                _ensure_tts_container_running()
                _wait_for_tts_ready()
                self.tts_active = True
            except Exception as e:
                logger.error(f"[SmartResourceManager] TTS container failed to start: {e}")
                raise

            current_allocated = self.base_os_mb + self.tts_vram_mb

            return {
                "action": "unloaded_llm_for_tts",
                "llm_active": self.llm_active,
                "tts_active": self.tts_active,
                "estimated_ram_used_mb": current_allocated,
                "unified_ram_remaining_mb": self.vram_total_mb - current_allocated
            }

    async def prepare_for_llm(self) -> Dict[str, Any]:
        """
        Pauses the Kokoro TTS container to free up RAM/CPU prior to heavy LLM context processing.
        """
        async with self.lock:
            logger.info("[SmartResourceManager] Prepping memory for LLM inference...")

            # 1. Pause TTS Docker container
            try:
                safely_pause_tts()
                self.tts_active = False
            except Exception as e:
                logger.warning(f"[SmartResourceManager] TTS pause warning: {e}")

            self.llm_active = True
            current_allocated = self.base_os_mb + self.llm_vram_mb

            return {
                "action": "paused_tts_for_llm",
                "llm_active": self.llm_active,
                "tts_active": self.tts_active,
                "estimated_ram_used_mb": current_allocated,
                "unified_ram_remaining_mb": self.vram_total_mb - current_allocated
            }

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns real-time resource allocation telemetry for UI/Worker metrics."""
        allocated = self.base_os_mb
        if self.llm_active:
            allocated += self.llm_vram_mb
        if self.tts_active:
            allocated += self.tts_vram_mb

        return {
            "llm_active": self.llm_active,
            "tts_active": self.tts_active,
            "allocated_ram_gb": round(allocated / 1024, 2),
            "total_ram_gb": round(self.vram_total_mb / 1024, 2),
            "free_ram_gb": round(max(0, self.vram_total_mb - allocated) / 1024, 2),
        }


# Global Instance
resource_manager = SmartResourceManager(total_ram_mb=8192)



# --- Request Schemas ---
class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float = 0.7


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_bella"
    speed: float = 0.9
    response_format: str = "wav"


class AudiobookRequest(BaseModel):
    book_id: str
    chapter_id: str
    text_chunks: list[str]
    voice: str = "af_bella"


# --- Orchestration Helpers ---
def _docker_unavailable_detail() -> str:
    if docker_import_error is None:
        return "Docker is unavailable: No Docker SDK found and 'docker' CLI command failed."
    error_str = str(docker_import_error)
    detail = f"Docker is unavailable: {error_str}"
    if "distutils" in error_str.lower():
        detail += " (Note: Python requires 'pip install setuptools' for the Docker SDK)."
    return detail


def _run_docker_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
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
    TARGET_IP = "127.0.0.1"
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"http://{TARGET_IP}:8880/docs")
                if resp.status_code < 500:
                    print(f"[Orchestrator] TTS container ready (attempt {attempt + 1})")
                    return
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(delay)
    raise HTTPException(status_code=503, detail="TTS container failed to become ready after 30 seconds")


def _ensure_tts_container_running():
    if docker_client is not None and docker is not None:
        try:
            container = docker_client.containers.get(TTS_Container_name)
            if container.status == "paused":
                container.unpause()
            elif container.status != "running":
                container.start()
            return
        except docker_not_found_error:
            docker_client.containers.run(
                TTS_Container_image,
                name=TTS_Container_name,
                detach=True,
                ports={'8880/tcp': 8880},
                mem_limit="6g",
            )
            return
        except Exception as exc:
            print(f"[Orchestrator] Docker SDK failed, falling back to CLI: {exc}")

    try:
        inspect = _run_docker_cli(["inspect", "-f", "{{.State.Status}}", TTS_Container_name])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_docker_unavailable_detail()) from exc

    if inspect.returncode == 0:
        status_str = inspect.stdout.strip().lower()
        if status_str == "paused":
            _run_docker_cli(["unpause", TTS_Container_name])
        elif status_str != "running":
            _run_docker_cli(["start", TTS_Container_name])
        return

    _run_docker_cli([
        "run", "-d", "--name", TTS_Container_name, "-p", "8880:8880", "--memory=6g", TTS_Container_image,
    ])


def _pause_tts_container_cli():
    try:
        inspect = _run_docker_cli(["inspect", "-f", "{{.State.Status}}", TTS_Container_name])
    except FileNotFoundError:
        return
    if inspect.returncode == 0 and inspect.stdout.strip().lower() == "running":
        _run_docker_cli(["pause", TTS_Container_name])


def safely_pause_tts():
    if docker_client is None or docker is None:
        _pause_tts_container_cli()
        return
    try:
        container = docker_client.containers.get(TTS_Container_name)
        if container.status == "running":
            container.pause()
            time.sleep(0.5)
    except Exception:
        _pause_tts_container_cli()


async def safely_unload_llm():
    api_v1_url = LM_STUDIO_URL.replace("/v1", "/api/v1")
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            loaded_instances = []
            v1_resp = await client.get(f"{api_v1_url}/models")
            if v1_resp.status_code == 200:
                for m in v1_resp.json().get("models", []):
                    for instance in m.get("loaded_instances", []):
                        if instance.get("id"):
                            loaded_instances.append(instance["id"])
            for instance_id in loaded_instances:
                await client.post(f"{api_v1_url}/models/unload", json={"instance_id": instance_id})
        except Exception:
            pass


async def get_lm_studio_models() -> List[str]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{LM_STUDIO_URL}/models")
            if response.status_code == 200:
                return [model["id"] for model in response.json().get("data", [])]
            return []
        except Exception:
            return []


# --- MetaDefender Cloud Security Engine ---
async def scan_file_bytes_metadefender(file_bytes: bytes, filename: str) -> dict:
    """
    Lightweight scanner that offloads antivirus inspection to MetaDefender Cloud API.
    Performs fast in-memory signature and executable header validation first.
    """
    # 1. In-Memory EICAR Test Signature Check
    eicar_sig = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    if eicar_sig in file_bytes:
        return {"clean": False, "threatName": "EICAR-Test-File", "scanner": "In-Memory Heuristics"}

    # 2. Executable Header Validation (Detects .exe / Mach-O / ELF disguised as audio)
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext in ["mp3", "wav", "flac", "ogg", "m4a", "aac"]:
        if file_bytes.startswith(b"MZ") or file_bytes.startswith(b"\x7fELF") or file_bytes.startswith(
                (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe")):
            return {"clean": False, "threatName": "Disguised-Executable-Binary", "scanner": "Header-Validator"}

    # Fallback to local validation if API key is not configured
    if not METADEFENDER_API_KEY:
        return {"clean": True, "threatName": None, "scanner": "Header-Validator (MetaDefender Key Unset)"}

    # 3. Fast Hash Lookup (Checks if OPSWAT analyzed this exact binary previously)
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    async with httpx.AsyncClient(timeout=10.0) as client:
        headers = {"apikey": METADEFENDER_API_KEY}

        try:
            hash_resp = await client.get(f"{METADEFENDER_URL}/hash/{sha256_hash}", headers=headers)

            if hash_resp.status_code == 200:
                data = hash_resp.json()
                scan_results = data.get("scan_results", {})

                progress = scan_results.get("progress_percentage", 0)
                scan_all_result_a = scan_results.get("scan_all_result_a", "")

                if progress == 100:
                    if scan_all_result_a in ["No threat detected", "Clean"]:
                        return {"clean": True, "threatName": None, "scanner": "MetaDefender Hash Lookup"}
                    elif scan_all_result_a in ["Infected", "Suspicious"]:
                        return {"clean": False, "threatName": "MetaDefender-Detected-Malware",
                                "scanner": "MetaDefender Hash Lookup"}

            # 4. Upload File Payload for Deep Cloud Scanning
            upload_headers = {**headers, "filename": filename}
            scan_resp = await client.post(f"{METADEFENDER_URL}/file", headers=upload_headers, content=file_bytes)

            if scan_resp.status_code == 200:
                data = scan_resp.json()
                data_id = data.get("data_id")

                for _ in range(3):
                    await asyncio.sleep(2)
                    poll_resp = await client.get(f"{METADEFENDER_URL}/file/{data_id}", headers=headers)
                    if poll_resp.status_code == 200:
                        poll_data = poll_resp.json()
                        scan_results = poll_data.get("scan_results", {})
                        if scan_results.get("progress_percentage") == 100:
                            is_clean = scan_results.get("scan_all_result_a") in ["No threat detected", "Clean"]
                            threat = None if is_clean else "MetaDefender-Flagged-Payload"
                            return {"clean": is_clean, "threatName": threat, "scanner": "MetaDefender Cloud Scanner"}

        except Exception as e:
            print(f"[MetaDefender Error] API call failed: {e}. Defaulting to header validation.")

    return {"clean": True, "threatName": None, "scanner": "Header-Validator (MetaDefender Fallback)"}


# --- Web Portal Views ---
@app.get("/")
async def serve_portal():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "index.html not found."})


@app.get("/reader")
async def serve_novel_reader():
    file_path = os.path.join(os.path.dirname(__file__), "WebNovelReader.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "WebNovelReader.html not found."})

@app.get("/chat")
async def serve_Chat():
    file_path = os.path.join(os.path.dirname(__file__), "chat.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "chat.html not found."})


# --- AI Gateway Endpoints ---
@app.get("/v1/models")
async def list_models():
    model_ids = await get_lm_studio_models()
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "lm-studio"} for m in model_ids],
    }


@app.post("/v1/chat/completions")
async def handle_chat(payload: ChatRequest):
    if not payload.model or payload.model.lower() == "default":
        raise HTTPException(status_code=400, detail="No model specified.")

    # Manage memory allocation
    await resource_manager.prepare_for_llm()

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{LM_STUDIO_URL}/chat/completions", json=payload.model_dump())
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"LM Studio Error: {str(e)}")


@app.post("/v1/audio/speech")
async def handle_speech(payload: SpeechRequest):
    # Manage memory allocation
    await resource_manager.prepare_for_tts()

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            tts_response = await client.post(
                "http://127.0.0.1:8880/v1/audio/speech",
                json=payload.model_dump()
            )
            if tts_response.status_code != 200:
                raise HTTPException(
                    status_code=tts_response.status_code,
                    detail=f"Kokoro Engine Rejected Request: {tts_response.text}"
                )
            return Response(content=tts_response.content, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")


@app.get("/v1/telemetry")
async def get_telemetry():
    """Exposes real-time memory telemetry to WebNovelReader dashboard."""
    return resource_manager.get_telemetry()

# --- Background Audiobook Generation ---
async def process_audiobook_background(job_id: str, payload: AudiobookRequest):
    audiobook_jobs[job_id] = {"status": "processing", "progress": 0}
    combined_audio = bytearray()

    try:
        _ensure_tts_container_running()
        _wait_for_tts_ready()
        total_chunks = len(payload.text_chunks)

        async with httpx.AsyncClient(timeout=180.0) as client:
            for index, chunk in enumerate(payload.text_chunks):
                tts_resp = await client.post(
                    "http://127.0.0.1:8880/v1/audio/speech",
                    json={"model": "kokoro", "input": chunk, "voice": payload.voice, "response_format": "wav"}
                )
                if tts_resp.status_code == 200:
                    combined_audio.extend(tts_resp.content)

                audiobook_jobs[job_id]["progress"] = int(((index + 1) / total_chunks) * 100)

        filename = f"audiobooks/{payload.book_id}_chap_{payload.chapter_id}.wav"
        await upload_audio_to_r2(bytes(combined_audio), filename, content_type="audio/wav")

        audiobook_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "r2_key": filename
        }
    except Exception as e:
        print(f"[Audiobook Job {job_id}] Error: {e}")
        audiobook_jobs[job_id] = {"status": "failed", "error": str(e)}


@app.post("/v1/audiobook/generate")
async def start_audiobook_generation(payload: AudiobookRequest):
    job_id = await exporter.create_export_job(
        book_id=payload.book_id,
        chapter_id=payload.chapter_id,
        title=f"Book {payload.book_id} - Chap {payload.chapter_id}",
        chunks=payload.text_chunks,
        voice=payload.voice
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/audiobook/status/{job_id}")
async def check_audiobook_status(job_id: str):
    job = exporter.get_job_status(job_id)
    if job.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# --- Storage & Security Endpoints ---
@app.post("/v1/storage/scan")
async def scan_uploaded_file(file: UploadFile = File(...)):
    """Scans an uploaded file payload using MetaDefender Cloud."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload.")

    filename = file.filename or "unnamed_file"
    scan_res = await scan_file_bytes_metadefender(file_bytes, filename)

    return {
        "filename": filename,
        "size": len(file_bytes),
        "clean": scan_res["clean"],
        "threatName": scan_res.get("threatName"),
        "scanner": scan_res.get("scanner"),
    }


@app.post("/v1/storage/upload")
async def upload_and_scan_file(file: UploadFile = File(...)):
    """Scans file for malware via MetaDefender. If clean, stores it in R2 automatically."""
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload.")

    filename = file.filename or "uploaded_audio.wav"
    scan_res = await scan_file_bytes_metadefender(file_bytes, filename)

    if not scan_res["clean"]:
        raise HTTPException(
            status_code=403,
            detail=f"Security scan rejected file: {scan_res.get('threatName', 'Malware detected')}"
        )

    ext = filename.split(".")[-1].lower() if "." in filename else "wav"
    media_type = RESPONSE_FORMAT_MEDIA_TYPES.get(ext, "audio/wav")

    success = await upload_audio_to_r2(file_bytes, filename, content_type=media_type)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to upload scanned file to R2 storage.")

    return {
        "message": "File successfully scanned and uploaded to R2.",
        "filename": filename,
        "size": len(file_bytes),
        "clean": True,
        "scanner": scan_res.get("scanner"),
    }


@app.get("/v1/storage/list")
async def list_r2_files():
    """Returns a list of all audio files buffered in R2."""
    files = get_r2_storage_list()
    return {"bucket": os.getenv("R2_BUCKET_NAME", "audio-cacha"), "files": files}


@app.get("/v1/storage/download/{filename:path}")
async def download_r2_file(filename: str):
    """Downloads an audio file directly from R2 through the FastAPI gateway."""
    file_bytes = download_file_from_r2(filename)
    if file_bytes is None:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in R2 storage.")

    ext = filename.split(".")[-1].lower() if "." in filename else "wav"
    media_type = RESPONSE_FORMAT_MEDIA_TYPES.get(ext, "application/octet-stream")
    return Response(content=file_bytes, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)