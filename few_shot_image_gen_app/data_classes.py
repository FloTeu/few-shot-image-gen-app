from few_shot_image_gen_app.selenium_fns import SeleniumBrowser
from few_shot_image_gen_app.llm_output import ImagePromptOutputModel
from llm_prompting_gen.generators import ParsablePromptEngineeringGenerator
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from PIL import Image as PILImage

class CrawlingTargetPage(str, Enum):
    MIDJOURNEY = "midjourney.com"
    OPENART = "openart.ai"

class SearchByCrawling(str, Enum):
    PROMPT = "Prompt"
    IMAGE_SIMILARITY = "Image Similarity"

class ImageModelCrawling(str, Enum):
    STABLE_DIFFUSION = "Stable Diffusion"
    MIDJOURNEY = "Midjourney"
    DALLE_2 = "DALL·E 2"

class ImageModelGeneration(str, Enum):
    STABLE_DIFFUSION = "Stable Diffusion (SDXL)"
    STABLE_DIFFUSION_CUSTOM_LORA = "Stable Diffusion LoRa"
    STABLE_DIFFUSION_CUSTOM_REPLICATE = "Stable Diffusion Replicate Training"
    DALLE_3 = "DALL-E 3"


class PromptGenerationModel(str, Enum):
    GPT_35 = "GPT 3-5 (ChatGPT)"
    GPT_4 = "GPT 4"

@dataclass
class AIImage:
    image_url: str
    prompt: str

@dataclass
class CrawlingRequest:
    search_term: str
    image_ais: List[ImageModelCrawling]

@dataclass
class CrawlingData:
    images: List[AIImage] = field(default_factory=list)  # crawled gen AI images

@dataclass
class Status:
    midjourney_login: bool = False
    page_crawled: bool = False
    prompts_generated: bool = False

@dataclass
class ImagePromptPair:
    image_pil: PILImage.Image
    prompt: str

@dataclass
class ImageGenerationData:
    gen_image_prompt_list: List[ImagePromptPair] | None = None
    prompt_gen_llm_output: ImagePromptOutputModel | None = None
    prompt_generator: ParsablePromptEngineeringGenerator | None = None

@dataclass
class SessionState:
    crawling_request: CrawlingRequest
    browser: SeleniumBrowser
    crawling_data: CrawlingData
    image_generation_data: ImageGenerationData
    status: Status
    session_id: str