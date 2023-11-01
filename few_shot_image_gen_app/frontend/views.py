import math
import json
from typing import List

import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI

from llm_few_shot_gen.generators import ParsablePromptEngineeringGenerator
from llm_few_shot_gen.models.output import ImagePromptOutputModel
from llm_few_shot_gen.models.prompt_engineering import PEFewShotExample
from few_shot_image_gen_app.data_classes import AIImage, SessionState, ImageModelGeneration
from few_shot_image_gen_app.image.generate import generate_with_stable_diffusion
from few_shot_image_gen_app.utils import extract_json_from_text


MAX_IMAGES_PER_ROW = 4

def split_list(list_obj, split_size):
    return [list_obj[i:i+split_size] for i in range(0, len(list_obj), split_size)]


def display_crawled_ai_images(ai_images: List[AIImage], make_collapsable=False):
    """ Displays already crawled midjourney images with prompts to frontend.
    """

    expander = st.expander("Collapse Midjourney images", expanded=True) if make_collapsable else st
    progress_text = "Display crawling results..."
    crawling_progress_bar = expander.progress(89, text=progress_text)
    display_images = expander.empty()
    display_cols = display_images.columns(MAX_IMAGES_PER_ROW)
    for j, midjourney_images_splitted_list in enumerate(split_list(ai_images, MAX_IMAGES_PER_ROW)):
        for i, midjourney_image in enumerate(midjourney_images_splitted_list):
            crawling_progress_bar.progress(math.ceil(89 + (10 / len(ai_images) * ((j * MAX_IMAGES_PER_ROW) + i)) + 1),
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
        # Images
        st.subheader("Selected Midjourney Images")
        selected_prompts: List[int] = st.session_state["selected_prompts"]
        selected_midjourney_images = [mid_img for i, mid_img in enumerate(midjourney_images) if
                                      (i + 1) in selected_prompts]
        display_crawled_ai_images(selected_midjourney_images, make_collapsable=True)

        # Prompt Engineering
        st.subheader("Prompt Engineering")
        expander = st.expander("Click to see details about prompt generation", expanded=False)
        prompt_generator = session_state.image_generation_data.prompt_generator
        with expander:
            markdown = "# Prompt for Text-to-Image Prompt Generation\n"
            markdown2 = ""
            markdown += "## 1. Role\n"
            markdown += prompt_generator.prompt_elements.role.replace("helpful assistant", ":green[helpful assistant]")
            markdown += "\n"
            markdown += "## 2. Instruction\n"
            markdown += prompt_generator.prompt_elements.instruction.replace("create text-to-image prompts", ":green[create text-to-image prompts]")
            markdown += "\n"
            markdown += "## 3. Context\n"
            markdown += prompt_generator.prompt_elements.context.replace("detailed and specific", ":green[detailed and specific]").replace("categories", ":green[categories]")
            markdown += "\n"
            markdown += "## 4. Output\n"
            pydantic_format = extract_json_from_text(prompt_generator.prompt_elements.output_format)
            markdown += prompt_generator.prompt_elements.output_format.replace("JSON schema", ":green[JSON schema]").replace(json.dumps(pydantic_format), "").replace("```","")
            markdown2 += "## 5. Few Shot Examples\n"
            if prompt_generator.prompt_elements.examples.intro:
                markdown2 += prompt_generator.prompt_elements.examples.intro.replace("underlying format", ":green[underlying format]")
                markdown2 += "\n"
            for example in prompt_generator.prompt_elements.examples.human_ai_interaction:
                if example.human:
                    markdown2 += ("Human: " + example.human)
                markdown2 += "\n"
                markdown2 += (f"AI: :orange[{example.ai}]")
                markdown2 += "\n\n"
            markdown2 += "## 6. Input\n"
            markdown2 += prompt_generator.prompt_elements.input.replace("{text}",f":orange[{st.session_state['prompt_gen_input']}]").replace("tasks", ":green[tasks]").replace("five concise english prompts", ":green[five concise english prompts]").replace("overarching styles or artists", ":green[overarching styles or artists]").replace("include your found styles or artists of step 1", ":green[include your found styles or artists of step 1]")
            markdown2 += "\n"
            # Display Markdown
            st.markdown(markdown)
            st.write(pydantic_format)
            st.markdown(markdown2)


def generate_image_model_prompts(prompts: List[str], tab_prompt_gen) -> ImagePromptOutputModel:
    print("Generate image prompts with few shots", prompts)
    with tab_prompt_gen:
        with st.spinner('Wait for prompt generation'):
            llm = ChatOpenAI(temperature=st.session_state["temperature"], model_name="gpt-3.5-turbo")
            prompt_gen = ParsablePromptEngineeringGenerator.from_json("templates/stable_diffusion_prompt_gen.json", llm=llm, pydantic_cls=ImagePromptOutputModel)
            # Overwrite few shot examples
            human_ai_interaction = []
            for prompt in prompts:
                human_ai_interaction.append(PEFewShotExample(ai=prompt))
            prompt_gen.prompt_elements.examples.human_ai_interaction = human_ai_interaction
            llm_output = prompt_gen.generate(text=st.session_state["prompt_gen_input"])
    # store results into session object
    session_state: SessionState = st.session_state["session_state"]
    session_state.image_generation_data.prompt_generator = prompt_gen
    session_state.image_generation_data.prompt_gen_llm_output = llm_output
    return llm_output


def display_image_gen_tab():
    session_state: SessionState = st.session_state["session_state"]

    st.subheader("Image Generation Prompt")
    llm_output: ImagePromptOutputModel | None = session_state.image_generation_data.prompt_gen_llm_output
    if llm_output:
        prompt_suggestion = st.selectbox("Generated Prompts", llm_output.image_prompts)
        prompt = st.text_area("Prompt", value=prompt_suggestion)
        # Atm only SDXL is available
        image_ai_model = st.selectbox("Image GenAI Model", (ImageModelGeneration.STABLE_DIFFUSION.value, ""))
        if st.button("Generate Image", key="Image Gen Button"):
            with st.spinner('Image generation...'):
                try:
                    image = generate_with_stable_diffusion(prompt)
                except Exception as e:
                    print(str(e))
                    st.warning("Something went wrong during image generation. Please try again.")
                session_state.image_generation_data.gen_image_pil = image
        else:
            image: Image | None = session_state.image_generation_data.gen_image_pil

        if image:
            st.image(image, width=512)
