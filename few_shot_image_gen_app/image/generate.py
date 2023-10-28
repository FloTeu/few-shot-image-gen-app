import replicate
import requests

from enum import Enum
from PIL import Image

from few_shot_image_gen_app.image.conversion import bytes2pil


class OutputFormat(Enum):
    STRING="str"
    GENERATOR="generator"


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
    model = "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316"
    return replicate_generate(model, {"prompt": prompt, "apply_watermark": False}, output_format=OutputFormat.GENERATOR)
