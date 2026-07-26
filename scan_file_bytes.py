import httpx
import asyncio

BASE_URL = "http://localhost:8000"


async def test_scans():
    async with httpx.AsyncClient(timeout=10.0) as client:
        print("=== GATE8 SCANNER TEST SUITE ===\n")

        # 1. Test Valid WAV File Payload
        # Dummy minimal valid WAV header
        valid_wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        files = {"file": ("test_sample.wav", valid_wav_bytes, "audio/wav")}

        resp = await client.post(f"{BASE_URL}/v1/storage/scan", files=files)
        print("1. Valid WAV Test:")
        print(f"   Status Code: {resp.status_code}")
        print(f"   Response: {resp.json()}\n")

        # 2. Test EICAR Anti-Virus Test Signature
        eicar_payload = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
        files = {"file": ("eicar_test.wav", eicar_payload, "audio/wav")}

        resp = await client.post(f"{BASE_URL}/v1/storage/scan", files=files)
        print("2. EICAR Signature Test:")
        print(f"   Status Code: {resp.status_code}")
        print(f"   Response: {resp.json()}\n")

        # 3. Test Disguised Executable Header Check (DOS executable 'MZ' disguised as .wav)
        disguised_exe_payload = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
        files = {"file": ("fake_audio.wav", disguised_exe_payload, "audio/wav")}

        resp = await client.post(f"{BASE_URL}/v1/storage/scan", files=files)
        print("3. Disguised Binary Header Test:")
        print(f"   Status Code: {resp.status_code}")
        print(f"   Response: {resp.json()}\n")


if __name__ == "__main__":
    asyncio.run(test_scans())