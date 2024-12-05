from pathlib import Path
from urllib.parse import urljoin

import aiofiles
from decouple import config
from fastapi import UploadFile
from pydantic import HttpUrl

from similarities.models import Image

IMAGES_DIRECTORY = "uploaded_images"


async def save_uploaded_file(unique_id: str, image: UploadFile) -> Path:
    base_dir = Path(config("STORAGE_DIR"))
    extension = Path(image.filename).suffix
    filename = f'{unique_id}{extension}'
    destination_directory = base_dir / IMAGES_DIRECTORY / unique_id[:2] / unique_id[2:4]
    destination_directory.mkdir(parents=True, exist_ok=True)

    destination_path = destination_directory / filename
    async with aiofiles.open(destination_path, "wb") as out_file:
        content = await image.read()
        await out_file.write(content)

    return destination_path


def get_image_public_url(image: Image) -> HttpUrl:
    return HttpUrl(urljoin(config("SERVICE_URL"), image.path))
