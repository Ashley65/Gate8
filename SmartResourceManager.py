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