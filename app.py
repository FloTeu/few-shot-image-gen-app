import streamlit as st
import streamlit.web.bootstrap as st_bootstrap

import os
import math
import requests
from typing import List
from PIL import Image
from io import BytesIO
from few_shot_image_gen_app.session import update_request
from few_shot_image_gen_app.utils import init_environment
from few_shot_image_gen_app.data_classes import SessionState, ImageModelCrawling, ImageModelGeneration, AIImage, CrawlingTargetPage
from few_shot_image_gen_app.crawling.midjourney import crawl_midjourney, login_to_midjourney
from few_shot_image_gen_app.crawling.openart_ai import crawl_openartai, crawl_openartai_similar_images
from few_shot_image_gen_app.image.generate import generate_with_stable_diffusion
from llm_few_shot_gen.generators import MidjourneyPromptGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel
from langchain.chat_models.openai import ChatOpenAI

# Add secrets to environment
init_environment()

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

def display_midjourney_images(midjourney_images: List[AIImage], make_collapsable=False):
    """ Displays already crawled midjourney images with prompts to frontend.
    """

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


def display_prompt_generation_tab(midjourney_images):
    session_state: SessionState = st.session_state["session_state"]
    llm_output: ImagePromptOutputModel = session_state.image_generation_data.prompt_gen_llm_output

    st.subheader("Generated Prompts")
    #if tab_prompt_gen.button("Regenerate Prompt"):
    #    llm_output = generate_midjourney_prompts(prompts)
    #print("llm_output", llm_output)
    st.write(llm_output.image_prompts)
    st.subheader("Detected Art Styles")
    st.write(llm_output.few_shot_styles)
    st.subheader("Detected Artists")
    st.write(llm_output.few_shot_artists)

    # Display selected images/prompts
    if "selected_prompts" in st.session_state:
        st.subheader("Selected Midjourney Images")
        selected_prompts: List[int] = st.session_state["selected_prompts"]
        selected_midjourney_images = [mid_img for i, mid_img in enumerate(midjourney_images) if
                                      (i + 1) in selected_prompts]
        display_midjourney_images(selected_midjourney_images, make_collapsable=True)

def generate_image_model_prompts(prompts: List[str]) -> ImagePromptOutputModel:
    with st.spinner('Wait for prompt generation'):
        llm = ChatOpenAI(temperature=st.session_state["temperature"], model_name="gpt-3.5-turbo")
        midjourney_prompt_gen = MidjourneyPromptGenerator(llm, pydantic_cls=ImagePromptOutputModel)
        midjourney_prompt_gen.set_few_shot_examples(prompts)
        llm_output = midjourney_prompt_gen.generate(text=st.session_state["prompt_gen_input"])
    # store results into session object
    session_state: SessionState = st.session_state["session_state"]
    session_state.image_generation_data.prompt_gen_llm_output = llm_output
    return llm_output

def display_image_gen_tab():
    session_state: SessionState = st.session_state["session_state"]

    st.subheader("Image Generation Prompt")
    llm_output: ImagePromptOutputModel | None = session_state.image_generation_data.prompt_gen_llm_output
    if llm_output:
        prompt = st.text_area("Prompt", value=llm_output.image_prompts[0])
        # Atm only SDXL is available
        image_ai_model = st.selectbox("Image GenAI Model", (ImageModelGeneration.STABLE_DIFFUSION.value, ""))
        if st.button("Generate Image", key="Image Gen Button"):
            with st.spinner('Wait for image generation'):
                image = generate_with_stable_diffusion(prompt)
                session_state.image_generation_data.gen_image_pil = image
        else:
            image: Image | None = session_state.image_generation_data.gen_image_pil

        if image:
            st.image(image)


def main():

    st.title("Image Gen AI Prompt Generator")
    st.caption('“If you can imagine it, you can generate it” - Runway Gen-2 commercial')

    st.write("Streamlit application for a showcase of the [LLM Few Shot Generator Library](https://github.com/FloTeu/llm-few-shot-generator). \n"
             "The app allows you to extract sample prompts from the Midjourney website. A subsample of these prompts can then be used to generate new prompts for ChatGPT using a [few-shot learning](https://www.promptingguide.ai/techniques/fewshot) approach.")
    st.write("[Source code frontend](https://github.com/FloTeu/few-shot-image-gen-app)")
    st.write("[Source code backend](https://github.com/FloTeu/llm-few-shot-generator)")

    with st.expander("Example"):
        st.write("""
            Text Prompt Input: "Grandma" \n
            Midjourney Prompt Generator output images:
        """)
        st.image("assets/grandmas.jpg")

    tab_crawling, tab_prompt_gen, tab_image_gen = st.tabs(["Crawling", "Prompt Generation", "Image Generation"])
    if "session_state" in st.session_state:
        session_state: SessionState = st.session_state["session_state"]
        with tab_crawling:
            display_midjourney_images(session_state.crawling_data.images, make_collapsable=False)
        with tab_prompt_gen:
            if session_state.image_generation_data.prompt_gen_llm_output:
                display_prompt_generation_tab(session_state.crawling_data.images)
        with tab_image_gen:
            display_image_gen_tab()

    st.sidebar.subheader("1. Crawling Target Page")
    target_page: CrawlingTargetPage = st.sidebar.selectbox("Crawling target page", options=[CrawlingTargetPage.OPENART.value])#, CrawlingTargetPage.MIDJOURNEY.value])
    if target_page == CrawlingTargetPage.MIDJOURNEY:
        st.sidebar.info("*prompt search is only available for authenticated midjourney users")
        st.sidebar.text_input("Midjourney Email", value=os.environ.get("user_name", ""), key="mid_email")
        st.sidebar.text_input("Midjourney Password", type="password", value=os.environ.get("password", ""), key="mid_password")
        st.sidebar.button("Login", on_click=login_to_midjourney, key="button_midjourney_login")

    st.sidebar.subheader("2. Crawling")
    st.sidebar.multiselect("Image AI Target Prompts", [ImageModelCrawling.STABLE_DIFFUSION.value, ImageModelCrawling.MIDJOURNEY.value, ImageModelCrawling.DALLE_2.value], default=[ImageModelCrawling.STABLE_DIFFUSION.value], key="image_models", on_change=update_request)
    st.sidebar.text_input("Search Term (e.g. art style)", key="search_term", on_change=update_request)
    if st.sidebar.button("Start Crawling", on_click=crawl_openartai if target_page == CrawlingTargetPage.OPENART else crawl_midjourney, args=(tab_crawling, ), key="button_midjourney_crawling"):
        session_state: SessionState = st.session_state["session_state"]
        display_midjourney_images(session_state.crawling_data.images, make_collapsable=False)
        tab_crawling.info('Please go to "Prompt Generation" tab')

    # Crawl similar images
    if target_page == CrawlingTargetPage.OPENART and "session_state" in st.session_state and len(st.session_state["session_state"].crawling_data.images) > 0:
        session_state: SessionState = st.session_state["session_state"]
        deep_crawl_image_nr = st.sidebar.selectbox("(optional) Crawl Similar Images",
                                                   [i + 1 for i in range(len(session_state.crawling_data.images))], on_change=display_midjourney_images, args=(session_state.crawling_data.images, tab_crawling, False,))
        if st.sidebar.button("Start Similar Image Crawling", on_click=crawl_openartai_similar_images, args=(tab_crawling, deep_crawl_image_nr - 1, ), key="button_midjourney_crawling_similar_images"):
            display_midjourney_images(session_state.crawling_data.images, make_collapsable=False)

    if "session_state" in st.session_state:
        session_state: SessionState = st.session_state["session_state"]
        st.sidebar.subheader("3. Prompt Generation")
        ai_images = session_state.crawling_data.images
        st.sidebar.number_input("LLM Temperature", value=0.7, max_value=1.0, min_value=0.0, key="temperature")
        selected_prompts = st.sidebar.multiselect("Select Designs for prompt generation:", [i+1 for i in range(len(ai_images))], key='selected_prompts')
        prompts = [mid_img.prompt for i, mid_img in enumerate(ai_images) if (i + 1) in selected_prompts]
        st.sidebar.text_input("Prompt Gen Input", on_change=generate_image_model_prompts, args=(prompts, ), key="prompt_gen_input")
        st.sidebar.button("Prompt Generation", key="button_prompt_generation")

if __name__ == "__main__":
    main()
