import math
from typing import List

import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from llm_few_shot_gen.generators import MidjourneyPromptGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel

from few_shot_image_gen_app.data_classes import AIImage, SessionState, ImageModelGeneration
from few_shot_image_gen_app.image.generate import generate_with_stable_diffusion


MAX_IMAGES_PER_ROW = 4

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
