import streamlit as st
import os, sys
import math
import requests
from typing import List
from io import BytesIO
from utils.session import update_request, SessionState
from utils.crawling.midjourney import crawl_midjourney, login_to_midjourney
from utils.crawling.openart_ai import crawl_openartai, crawl_openartai_similar_images
from utils.data_classes import MidjourneyImage, CrawlingTargetPage
from llm_few_shot_gen.generators import MidjourneyPromptGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel
from langchain.chat_models.openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["open_ai_api_key"]

MAX_IMAGES_PER_ROW = 4

st.set_page_config(
    page_title="Midjourney Prompt Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def image_url2image_bytes_io(image_url: str) -> BytesIO:
    response = requests.get(image_url)
    return BytesIO(response.content)

def split_list(list_obj, split_size):
    return [list_obj[i:i+split_size] for i in range(0, len(list_obj), split_size)]

def display_midjourney_images(midjourney_images: List[MidjourneyImage], tab, make_collapsable=False):
    """ Displays already crawled midjourney images with prompts to frontend.
    """

    with tab:
        expander = st.expander("Collapse Midjourney images", expanded=True) if make_collapsable else st
        progress_text = "Display crawling results..."
        crawling_progress_bar = expander.progress(89, text=progress_text)
        display_images = expander.empty()
        display_cols = display_images.columns(MAX_IMAGES_PER_ROW)
        for j, midjourney_images_splitted_list in enumerate(split_list(midjourney_images, MAX_IMAGES_PER_ROW)):
            for i, midjourney_image in enumerate(midjourney_images_splitted_list):
                crawling_progress_bar.progress(math.ceil(89 + (10 / len(midjourney_images) * ((j * MAX_IMAGES_PER_ROW) + i)) + 1),
                                               text=progress_text)
                #image_bytes_io: BytesIO = image_url2image_bytes_io(midjourney_image.image_url)
                #display_cols[i].image(image_bytes_io)
                display_cols[i].image(midjourney_image.image_url)
                #color = "black" if not midjourney_image.selected else "green"
                #display_cols[i].markdown(f":{color}[{(j * MAX_IMAGES_PER_ROW) + i + 1}. {mba_product.title}]")
                display_cols[i].write(f"{(j * MAX_IMAGES_PER_ROW) + i + 1}: {midjourney_image.prompt}")

        crawling_progress_bar.empty()


def display_prompt_generation_tab(midjourney_images, selected_prompts, tab_prompt_gen, tab_crawling):
    # make sure crawling page is displayed as well
    display_midjourney_images(midjourney_images, tab_crawling, make_collapsable=False)

    # Few Shot learning
    prompts = [mid_img.prompt for i, mid_img in enumerate(midjourney_images) if (i + 1) in selected_prompts]

    with tab_prompt_gen:
        with st.spinner('Wait for prompt generation'):
            llm_output: ImagePromptOutputModel = generate_midjourney_prompts(prompts)

    tab_prompt_gen.subheader("Generated Prompts")
    #if tab_prompt_gen.button("Regenerate Prompt"):
    #    llm_output = generate_midjourney_prompts(prompts)
    #print("llm_output", llm_output)
    tab_prompt_gen.write(llm_output.image_prompts)
    tab_prompt_gen.subheader("Detected Art Styles")
    tab_prompt_gen.write(llm_output.few_shot_styles)
    tab_prompt_gen.subheader("Detected Artists")
    tab_prompt_gen.write(llm_output.few_shot_artists)

    # Display selected images/prompts
    tab_prompt_gen.subheader("Selected Midjourney Images")
    selected_midjourney_images = [mid_img for i, mid_img in enumerate(midjourney_images) if
                                  (i + 1) in selected_prompts]
    display_midjourney_images(selected_midjourney_images, tab_prompt_gen, make_collapsable=True)

def generate_midjourney_prompts(prompts) -> ImagePromptOutputModel:
    llm = ChatOpenAI(temperature=st.session_state["temperature"], model_name="gpt-3.5-turbo")
    midjourney_prompt_gen = MidjourneyPromptGenerator(llm, pydantic_cls=ImagePromptOutputModel)
    midjourney_prompt_gen.set_few_shot_examples(prompts)
    llm_output = midjourney_prompt_gen.generate(text=st.session_state["prompt_gen_input"])
    return llm_output


def main():

    st.title("Midjourney Prompt Generator")
    st.caption('“If you can imagine it, you can generate it” - Runway Gen-2 commercial')

    st.write("Streamlit application for a showcase of the [LLM Few Shot Generator Library](https://github.com/FloTeu/llm-few-shot-generator). \n"
             "The app allows you to extract sample prompts from the Midjourney website. A subsample of these prompts can then be used to generate new prompts for ChatGPT using a [few-shot learning](https://www.promptingguide.ai/techniques/fewshot) approach.")
    st.write("[Source code frontend](https://github.com/FloTeu/midjourney-prompt-generator)")
    st.write("[Source code backend](https://github.com/FloTeu/llm-few-shot-generator)")

    with st.expander("Example"):
        st.write("""
            Text Prompt Input: "Grandma" \n
            Midjourney Prompt Generator output images:
        """)
        st.image("assets/grandmas.jpg")

    tab_crawling, tab_prompt_gen = st.tabs(["Crawling", "Prompt Generation"])

    st.sidebar.subheader("1. Crawling Target Page")
    target_page: CrawlingTargetPage = st.sidebar.selectbox("Crawling target page", options=["openart.ai", "midjourney.com"])
    if target_page == CrawlingTargetPage.MIDJOURNEY:
        st.sidebar.info("*prompt search is only available for authenticated midjourney users")
        st.sidebar.text_input("Midjourney Email", value=os.environ.get("user_name", ""), key="mid_email")
        st.sidebar.text_input("Midjourney Password", type="password", value=os.environ.get("password", ""), key="mid_password")
        st.sidebar.button("Login", on_click=login_to_midjourney, key="button_midjourney_login")

    st.sidebar.subheader("2. Midjourney Crawling")
    st.sidebar.text_input("Search Term (e.g. art style)", key="search_term", on_change=update_request)
    if st.sidebar.button("Start Crawling", on_click=crawl_openartai if target_page == CrawlingTargetPage.OPENART else crawl_midjourney, args=(tab_crawling, ), key="button_midjourney_crawling"):
        session_state: SessionState = st.session_state["session_state"]
        display_midjourney_images(session_state.crawling_data.midjourney_images, tab_crawling, make_collapsable=False)
        tab_crawling.info('Please go to "Prompt Generation" tab')

    # Crawl similar images
    if target_page == CrawlingTargetPage.OPENART and "session_state" in st.session_state and len(st.session_state["session_state"].crawling_data.midjourney_images) > 0:
        session_state: SessionState = st.session_state["session_state"]
        deep_crawl_image_nr = st.sidebar.selectbox("(optional) Crawl Similar Images",
                                          [i + 1 for i in range(len(session_state.crawling_data.midjourney_images))], on_change=display_midjourney_images, args=(session_state.crawling_data.midjourney_images, tab_crawling, False, ))
        if st.sidebar.button("Start Similar Image Crawling", on_click=crawl_openartai_similar_images, args=(tab_crawling, deep_crawl_image_nr - 1, ), key="button_midjourney_crawling_similar_images"):
            display_midjourney_images(session_state.crawling_data.midjourney_images, tab_crawling,
                                      make_collapsable=False)

    if "session_state" in st.session_state:
        session_state: SessionState = st.session_state["session_state"]
        st.sidebar.subheader("3. Prompt Generation")
        midjourney_images = session_state.crawling_data.midjourney_images
        st.sidebar.number_input("LLM Temperature", value=0.7, max_value=1.0, min_value=0.0, key="temperature")
        selected_prompts = st.sidebar.multiselect("Select Designs for prompt generation:", [i+1 for i in range(len(midjourney_images))], on_change=display_midjourney_images, args=(session_state.crawling_data.midjourney_images,tab_crawling,False,), key='selected_prompts')
        st.sidebar.text_input("Prompt Gen Input", key="prompt_gen_input")
        st.sidebar.button("Prompt Generation", on_click=display_prompt_generation_tab, args=(midjourney_images, selected_prompts, tab_prompt_gen, tab_crawling, ), key="button_prompt_generation")




if __name__ == "__main__":
    main()
