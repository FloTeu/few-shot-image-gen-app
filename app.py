import streamlit as st

from few_shot_image_gen_app.frontend.sidebar import display_sidebar
from few_shot_image_gen_app.frontend.views import display_crawled_ai_images, display_prompt_generation_tab, \
    display_image_gen_tab
from few_shot_image_gen_app.utils import init_environment
from few_shot_image_gen_app.data_classes import SessionState

# Add secrets to environment
init_environment()


st.set_page_config(
    page_title="Midjourney Prompt Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("Image Gen AI Prompt Generator")
    st.caption('“If you can imagine it, you can generate it” - Runway Gen-2 commercial')

    st.write("Streamlit application for a showcase of the [LLM Few Shot Generator Library](https://github.com/FloTeu/llm-few-shot-generator). \n"
             "The app allows you to extract sample prompts from openart.ai website. A subsample of these prompts can then be used to generate new prompts for ChatGPT using a [few-shot learning](https://www.promptingguide.ai/techniques/fewshot) approach.")
    st.write("[Source code frontend](https://github.com/FloTeu/few-shot-image-gen-app)")
    st.write("[Source code backend](https://github.com/FloTeu/llm-few-shot-generator)")

    with st.expander("Example"):
        st.write("""
            Text Prompt Input: "Grandma" \n
            Midjourney Prompt Generator output images:
        """)
        st.image("assets/grandmas.jpg")

    # Display tabs
    tab_crawling, tab_prompt_gen, tab_image_gen = st.tabs(["Crawling", "Prompt Generation", "Image Generation"])
    if "session_state" in st.session_state:
        session_state: SessionState = st.session_state["session_state"]
        with tab_crawling:
            display_crawled_ai_images(session_state.crawling_data.images, make_collapsable=False)
        with tab_prompt_gen:
            if session_state.image_generation_data.prompt_gen_llm_output:
                display_prompt_generation_tab(session_state.crawling_data.images)
        with tab_image_gen:
            display_image_gen_tab()

    # Display sidebar
    display_sidebar(tab_crawling, tab_prompt_gen)

if __name__ == "__main__":
    main()
