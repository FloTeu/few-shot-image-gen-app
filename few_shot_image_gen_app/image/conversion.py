from io import BytesIO
from PIL import Image

def bytes2bytes_io(image_bytes: bytes) -> BytesIO:
    return BytesIO(image_bytes)

def bytes2pil(image_bytes: bytes) -> Image:
    return Image.open(bytes2bytes_io(image_bytes))

def pil2bytes_io(img_pil: Image, format="PNG") -> BytesIO:
    img_byte_arr = BytesIO()
    img_pil.save(img_byte_arr, format=format)
    return img_byte_arr

