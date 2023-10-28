from io import BytesIO
from PIL import Image

def bytes2bytes_io(image_bytes: bytes) -> BytesIO:
    return BytesIO(image_bytes)

def bytes2pil(image_bytes: bytes) -> Image:
    return Image.open(bytes2bytes_io(image_bytes))