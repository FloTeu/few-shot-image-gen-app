import os
import streamlit as st

from few_shot_image_gen_app.data_classes import SessionState, CrawlingRequest, CrawlingData, Status, ImageGenerationData
from few_shot_image_gen_app.selenium_fns import SeleniumBrowser


def booleanize(s):
    return s.lower() in ['true', '1']

def is_debug():
    return booleanize(os.environ.get("DEBUG", "False"))

def creat_session_state() -> SessionState:
    search_term = st.session_state["search_term"]
    image_ais = st.session_state["image_models"]
    request = CrawlingRequest(search_term=search_term, image_ais=image_ais)
    crawling_data = CrawlingData()
    status = Status()
    session_id = get_session_id()
    image_generation_data = ImageGenerationData()

    browser = SeleniumBrowser()
    browser.setup(headless=not is_debug())
    return SessionState(crawling_request=request, browser=browser, crawling_data=crawling_data, image_generation_data=image_generation_data, status=status, session_id=session_id)


def get_session_id():
    return st.runtime.scriptrunner.add_script_run_ctx().streamlit_script_run_ctx.session_id


def set_session_state_if_not_exists():
    """Creates a session state if its not already exists"""
    if "session_state" not in st.session_state:
        st.session_state["session_state"] = creat_session_state()


def update_request():
    set_session_state_if_not_exists()
    session_state: SessionState = st.session_state["session_state"]
    request = session_state.crawling_request
    request.search_term = st.session_state["search_term"]
    request.image_ais = st.session_state["image_models"]

    # Reset status
    session_state.status.page_crawled = False
    session_state.status.prompts_generated = False

