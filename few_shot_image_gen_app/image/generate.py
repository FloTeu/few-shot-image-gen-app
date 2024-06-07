import os
import asyncio
import replicate
import requests
import time
import openai
import streamlit as st
from typing import Coroutine, List

from io import BytesIO
from enum import Enum

from PIL import Image

from few_shot_image_gen_app.data_classes import ImageModelGeneration
from few_shot_image_gen_app.image.conversion import bytes2pil

SDXL_MODEL_VERSION = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
SDXL_LORA_MODEL_VERSION = "zylim0702/sdxl-lora-customize-model:5a2b1cff79a2cf60d2a498b424795a90e26b7a3992fbd13b340f73ff4942b81e"

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

def replicate_post(version: str, input: dict) -> str:
    """Returns id to get response afterwards with another get request"""
    headers = {'Content-type': 'application/json', "Authorization": f"Token {st.secrets['replicate']}"}
    url = "https://api.replicate.com/v1/predictions"
    body = {
        "version": version,
        "input": input
    }
    r = requests.post(url, json=body, headers=headers)
    return r.json()["id"]

async def async_replicate_run(model_version: str, input: dict):
    return replicate.async_run(model_version, input=input)


def generate_with_stable_diffusion(prompt: str) -> Image:
    return replicate_generate(SDXL_MODEL_VERSION, {"prompt": prompt, "apply_watermark": False}, output_format=OutputFormat.GENERATOR)


def generate_with_stable_diffusion_custom_lora(prompt: str, lora_url: str) -> Image:
    return replicate_generate(SDXL_LORA_MODEL_VERSION, {"prompt": prompt, "Lora_url": lora_url}, output_format=OutputFormat.GENERATOR)


def generate_with_stable_diffusion_custom_trained(prompt: str, model_version_url: str) -> Image:
    """Trained via replicate platform"""
    return replicate_generate(model_version_url, {"prompt": prompt}, output_format=OutputFormat.GENERATOR)


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

def generate_all_replicate(model_version: str, prompts: List[str], **input_kwargs) -> List[Image.Image]:
    response_ids = []
    for prompt in prompts:
        response_ids.append(replicate_post(#model=model_version.split(":")[0],
                       version=model_version.split(":")[1],
                       input={"prompt": prompt, **input_kwargs}))
    img_urls = []
    for response_id in response_ids:
        response = requests.get(f"https://api.replicate.com/v1/predictions/{response_id}",
                                headers={"Authorization": f"Token {st.secrets['replicate']}"})
        if response.json()["status"] == "succeeded":
            img_urls.append(response.json()["output"][0])
        while response.json()["status"] != "succeeded":
            response = requests.get(f"https://api.replicate.com/v1/predictions/{response_id}",
                                    headers={"Authorization": f"Token {st.secrets['replicate']}"})
            if response.json()["status"] in ["starting", "processing"]:
                # wait 2 secs
                time.sleep(2)
                continue
            elif response.json()["status"] == "succeeded":
                img_urls.append(response.json()["output"][0])
            else:
                print(f"Warning: id {response_id} could not be generated successfully")
                break
    return [bytes2pil(requests.get(img_url, stream=True).content) for img_url in img_urls]


def generate(prompts: List[str], image_ai_model: ImageModelGeneration, token_prefix=None, lora_tar_url=None, model_version_url=None) ->  List[Image.Image]:
    images: List[Image.Image] = []
    input_kwargs = {}
    if image_ai_model == ImageModelGeneration.STABLE_DIFFUSION:
        model_version = SDXL_MODEL_VERSION
    elif image_ai_model == ImageModelGeneration.STABLE_DIFFUSION_CUSTOM_LORA:
        model_version = SDXL_LORA_MODEL_VERSION
        prompts = [f"{token_prefix}{prompt}" for prompt in prompts]
        input_kwargs = {"Lora_url": lora_tar_url}
    elif image_ai_model == ImageModelGeneration.STABLE_DIFFUSION_CUSTOM_REPLICATE:
        model_version = model_version_url
        prompts = [f"{token_prefix}{prompt}" for prompt in prompts]
        input_kwargs = {}
    elif image_ai_model == ImageModelGeneration.DALLE_3:
        # Note: Dall-e is currently not implemented for async requests
        for prompt in prompts:
            images.append(generate_with_dalle3(prompt))
        return images
    else:
        raise NotImplementedError

    return generate_all_replicate(model_version=model_version, prompts=prompts, **input_kwargs)
