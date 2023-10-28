from few_shot_image_gen_app.selenium_fns import SeleniumBrowser
from llm_few_shot_gen.models.output import ImagePromptOutputModel
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from PIL import Image as PILImage

class CrawlingTargetPage(str, Enum):
    MIDJOURNEY = "midjourney.com"
    OPENART = "openart.ai"

class ImageModelCrawling(str, Enum):
    STABLE_DIFFUSION = "Stable Diffusion"
    MIDJOURNEY = "Midjourney"
    DALLE_2 = "DALL·E 2"

class ImageModelGeneration(str, Enum):
    STABLE_DIFFUSION = "Stable Diffusion (SDXL)"

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
class ImageGenerationData:
    gen_image_pil: PILImage.Image | None = None
    prompt_gen_llm_output: ImagePromptOutputModel | None = None

@dataclass
class SessionState:
    crawling_request: CrawlingRequest
    browser: SeleniumBrowser
    crawling_data: CrawlingData
    image_generation_data: ImageGenerationData
    status: Status
    session_id: str