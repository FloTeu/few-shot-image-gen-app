import os
import replicate
import requests
import openai

from io import BytesIO
from enum import Enum
from PIL import Image

from few_shot_image_gen_app.image.conversion import bytes2pil


class OutputFormat(Enum):
    STRING = "str"
    GENERATOR = "generator"


class OpenAIImageQuality(str, Enum):
    STANDARD = "standard"
    HD = "hd"


def replicate_generate(model_version: str, input: dict, output_format: OutputFormat = OutputFormat.STRING) -> Image:
    output = replicate.run(
        model_version,
        input=input
    )
    if output_format == OutputFormat.STRING:
        img_url = output
    if output_format == OutputFormat.GENERATOR:
        for output_i in output:
            img_url = output_i
    return bytes2pil(requests.get(img_url, stream=True).content)


def generate_with_stable_diffusion(prompt: str) -> Image:
    model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
    return replicate_generate(model, {"prompt": prompt, "apply_watermark": False}, output_format=OutputFormat.GENERATOR)


def generate_with_stable_diffusion_custom(prompt: str, lora_url: str) -> Image:
    model = "zylim0702/sdxl-lora-customize-model:5a2b1cff79a2cf60d2a498b424795a90e26b7a3992fbd13b340f73ff4942b81e"
    return replicate_generate(model, {"prompt": prompt, "Lora_url": lora_url}, output_format=OutputFormat.GENERATOR)


def generate_with_dalle3(prompt: str, quality: OpenAIImageQuality = OpenAIImageQuality.STANDARD) -> Image:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality=quality,
        n=1,
    )

    image_url = response.data[0].url
    response = requests.get(image_url)

    return Image.open(BytesIO(response.content))
