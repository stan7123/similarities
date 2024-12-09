from pathlib import Path


_images_dir = Path(__file__).parent.absolute() / "images"

IMAGES = {
    "apples": [
        _images_dir / "apple1.png",
        _images_dir / "apple2.png",
        _images_dir / "apple3.png",
        _images_dir / "apple1_small.png",
    ],
    "bananas": [
        _images_dir / "banana1.png",
        _images_dir / "banana2.png",
        _images_dir / "banana3.png",
    ],
    "kiwi": [
        _images_dir / "kiwi1.png",
        _images_dir / "kiwi2.png",
        _images_dir / "kiwi3.png",
    ],
}

for collection_name, collection in IMAGES.items():
    for path in collection:
        assert path.exists(), path
