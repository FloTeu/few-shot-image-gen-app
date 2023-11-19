import os

import streamlit as st

from few_shot_image_gen_app.frontend.views import display_crawled_ai_images, generate_image_model_prompts
from few_shot_image_gen_app.crawling.midjourney import login_to_midjourney, crawl_midjourney
from few_shot_image_gen_app.crawling.openart_ai import crawl_openartai, crawl_openartai_similar_images
from few_shot_image_gen_app.data_classes import CrawlingTargetPage, ImageModelCrawling, SessionState, PromptGenerationModel
from few_shot_image_gen_app.session import update_request


def display_sidebar(tab_crawling, tab_prompt_gen):
    st.sidebar.subheader("1. Crawling")
    # Target crawling page
    target_page: CrawlingTargetPage = st.sidebar.selectbox("Crawling target page", options=[
        CrawlingTargetPage.OPENART.value])  # , CrawlingTargetPage.MIDJOURNEY.value])
    if target_page == CrawlingTargetPage.MIDJOURNEY:
        st.sidebar.info("*prompt search is only available for authenticated midjourney users")
        st.sidebar.text_input("Midjourney Email", value=os.environ.get("user_name", ""), key="mid_email")
        st.sidebar.text_input("Midjourney Password", type="password", value=os.environ.get("password", ""),
                              key="mid_password")
        st.sidebar.button("Login", on_click=login_to_midjourney, key="button_midjourney_login")
    if target_page == CrawlingTargetPage.OPENART:
        st.sidebar.toggle("Community Only", value=False, help="Ensures image was uploaded with a community account", key="community_only")

    # Crawling Request
    st.sidebar.multiselect("Image AI Target Prompts",
                           [ImageModelCrawling.STABLE_DIFFUSION.value, ImageModelCrawling.MIDJOURNEY.value,
                            ImageModelCrawling.DALLE_2.value], default=[ImageModelCrawling.STABLE_DIFFUSION.value],
                           key="image_models", on_change=update_request)
    st.sidebar.text_input("Search Term (e.g. art style)", key="search_term", on_change=update_request)
    if st.sidebar.button("Start Crawling",
                         on_click=crawl_openartai if target_page == CrawlingTargetPage.OPENART else crawl_midjourney,
                         args=(tab_crawling,), key="button_midjourney_crawling"):
        session_state: SessionState = st.session_state["session_state"]
        display_crawled_ai_images(session_state.crawling_data.images, make_collapsable=False)
        tab_crawling.info('Please go to "Prompt Generation" tab')
    # Crawl similar images
    if target_page == CrawlingTargetPage.OPENART and "session_state" in st.session_state and len(
            st.session_state["session_state"].crawling_data.images) > 0:
        session_state: SessionState = st.session_state["session_state"]
        deep_crawl_image_nr = st.sidebar.selectbox("(optional) Crawl Similar Images",
                                                   [i + 1 for i in range(len(session_state.crawling_data.images))])
        st.sidebar.button("Start Similar Image Crawling", on_click=crawl_openartai_similar_images,
                             args=(tab_crawling, deep_crawl_image_nr - 1,),
                             key="button_midjourney_crawling_similar_images")

    if "session_state" in st.session_state:
        session_state: SessionState = st.session_state["session_state"]
        with st.sidebar:
            st.subheader("2. Prompt Generation")
            ai_images = session_state.crawling_data.images
            st.selectbox("LLM Model", (PromptGenerationModel.GPT_35.value, PromptGenerationModel.GPT_4.value), key="llm_model")
            st.number_input("LLM Temperature", value=0.7, max_value=1.0, min_value=0.0, key="temperature")
            selected_prompts = st.multiselect("Select Designs for prompt generation:",
                                                      [i + 1 for i in range(len(ai_images))], key='selected_prompts')
            prompts = [mid_img.prompt for i, mid_img in enumerate(ai_images) if (i + 1) in selected_prompts]
            st.text_input("Prompt Gen Input", key="prompt_gen_input")
            st.button("Prompt Generation", on_click=generate_image_model_prompts, args=(prompts, tab_prompt_gen,), key="button_prompt_generation")
