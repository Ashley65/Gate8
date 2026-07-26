
import os
import asyncio
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# Load environment variables from the .env file in the root folder
load_dotenv()

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "audio-cacha")

_r2_client = None


def get_r2_client():
    global _r2_client
    if _r2_client is None:
        if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
            print("[R2 Storage] Warning: R2 credentials incomplete. Remote uploads disabled.")
            return None

        endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        _r2_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto"
        )
    return _r2_client


def _sync_upload_bytes(file_bytes: bytes, filename: str, content_type: str = "audio/wav") -> bool:
    client = get_r2_client()
    if not client:
        return False
    try:
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=filename,
            Body=file_bytes,
            ContentType=content_type
        )
        print(f"[R2 Storage] Successfully buffered {filename} to bucket '{R2_BUCKET_NAME}'.")
        return True
    except (BotoCoreError, ClientError) as err:
        print(f"[R2 Storage] Upload failed for {filename}: {err}")
        return False


async def upload_audio_to_r2(file_bytes: bytes, filename: str, content_type: str = "audio/wav") -> bool:
    """Non-blocking async wrapper to upload bytes directly to Cloudflare R2."""
    return await asyncio.to_thread(_sync_upload_bytes, file_bytes, filename, content_type)


def generate_presigned_download_url(filename: str, expiration_seconds: int = 86400) -> str | None:
    """Generates a temporary public download URL for the app (Valid for 24 hours by default)."""
    client = get_r2_client()
    if not client:
        return None
    try:
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': R2_BUCKET_NAME, 'Key': filename},
            ExpiresIn=expiration_seconds
        )
        return url
    except (BotoCoreError, ClientError) as e:
        print(f"[R2 Storage] Presigned URL error: {e}")
        return None


def get_r2_storage_list() -> list[str]:
    """Returns a list of all files currently stored in the R2 bucket."""
    client = get_r2_client()
    if not client:
        return []
    try:
        response = client.list_objects_v2(Bucket=R2_BUCKET_NAME)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        else:
            return []
    except (BotoCoreError, ClientError) as e:
        print(f"[R2 Storage] List error: {e}")
        return []


def download_file_from_r2(filename: str) -> bytes | None:
    """Downloads a file from R2 and returns its bytes. Returns None if the file doesn't exist or on error."""
    client = get_r2_client()
    if not client:
        return None
    try:
        response = client.get_object(Bucket=R2_BUCKET_NAME, Key=filename)
        return response['Body'].read()
    except (BotoCoreError, ClientError) as e:
        print(f"[R2 Storage] Download error for {filename}: {e}")
        return None