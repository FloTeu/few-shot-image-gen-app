from few_shot_image_gen_app.selenium_fns import SeleniumBrowser
from dataclasses import dataclass, field
from typing import List
from enum import Enum

class CrawlingTargetPage(str, Enum):
    MIDJOURNEY = "midjourney.com"
    OPENART = "openart.ai"

class ImageModel(str, Enum):
    STABLE_DIFFUSION = "Stable Diffusion"
    MIDJOURNEY = "Midjourney"
    DALLE_2 = "DALL·E 2"


@dataclass
class AIImage:
    image_url: str
    prompt: str

@dataclass
class CrawlingRequest:
    search_term: str
    image_ais: List[ImageModel]

@dataclass
class CrawlingData:
    images: List[AIImage] = field(default_factory=list)  # crawled gen AI images

@dataclass
class Status:
    midjourney_login: bool = False
    page_crawled: bool = False
    prompts_generated: bool = False

@dataclass
class SessionState:
    crawling_request: CrawlingRequest
    browser: SeleniumBrowser
    crawling_data: CrawlingData
    status: Status
    session_id: str