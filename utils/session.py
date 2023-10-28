import os
import streamlit as st

from typing import List, Any
from utils.data_classes import SessionState, CrawlingRequest, CrawlingData, Status
from utils.selenium_fns import SeleniumBrowser


def booleanize(s):
    return s.lower() in ['true', '1']

def is_debug():
    return booleanize(os.environ.get("DEBUG", "False"))

def creat_session_state() -> SessionState:
    search_term = st.session_state["search_term"]
    request = CrawlingRequest(search_term=search_term)
    crawling_data = CrawlingData()
    status = Status()
    session_id = get_session_id()

    browser = SeleniumBrowser()
    browser.setup(headless=not is_debug())
    return SessionState(crawling_request=request, browser=browser, crawling_data=crawling_data, status=status, session_id=session_id)


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

    # Reset status
    session_state.status.page_crawled = False
    session_state.status.prompts_generated = False

