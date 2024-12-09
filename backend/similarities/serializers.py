from enum import Enum
from uuid import UUID

from pydantic import BaseModel, HttpUrl


class ImageCreationResponse(BaseModel):
    id: UUID


class SimilarResponseStatus(str, Enum):
    OK = "ok"
    PROCESSING = "processing"


class SimilarImageEntry(BaseModel):
    url: HttpUrl
    distance: float


class SimilarImagesResponse(BaseModel):
    status: SimilarResponseStatus
    image_url: HttpUrl
    similar_images: list[SimilarImageEntry]


class SearchType(str, Enum):
    COLORS = "colors"
    OBJECTS = "objects"
    TEXTURE = "texture"


SEARCH_TYPE_TO_COLUMN_NAME = {
    SearchType.COLORS: "color_hist",
    SearchType.OBJECTS: "hog_hist",
    SearchType.TEXTURE: "texture_hist",
}
