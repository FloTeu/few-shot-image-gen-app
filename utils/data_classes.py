from utils.selenium_fns import SeleniumBrowser
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class CrawlingTargetPage(str, Enum):
    MIDJOURNEY = "midjourney.com"
    OPENART = "openart.ai"

@dataclass
class MidjourneyImage:
    image_url: str
    prompt: str

@dataclass
class CrawlingRequest:
    search_term: str

@dataclass
class CrawlingData:
    midjourney_images: List[MidjourneyImage] = field(default_factory=list)  # crawled midjourney images

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