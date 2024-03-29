import time
import logging
import urllib.parse
import streamlit as st

from typing import List
from contextlib import suppress
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.keys import Keys

from few_shot_image_gen_app.crawling.selenium_fns import has_button_ancestor, wait_until_element_disappears, save_html
from few_shot_image_gen_app.session import set_session_state_if_not_exists
from few_shot_image_gen_app.data_classes import SessionState, CrawlingData, AIImage, ImageModelCrawling
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_discovery_url(search_term: str, image_models: List[ImageModelCrawling], only_community=False):
    search_type = "community-only" if only_community else "both"
    search_models = []
    for image_ai in image_models:
        if image_ai == ImageModelCrawling.STABLE_DIFFUSION:
            search_models.append("sd")
        if image_ai == ImageModelCrawling.DALLE_2:
            search_models.append("dalle2")
        if image_ai == ImageModelCrawling.MIDJOURNEY:
            search_models.append("md")

    method = "prompt"
    if "search_by" in st.session_state:
        method = "prompt" if st.session_state["search_by"] == "Prompt" else "similarity"

    # encode to url format
    search_models = urllib.parse.quote_plus(",".join(search_models))
    search_term = urllib.parse.quote_plus(search_term)
    return f"https://openart.ai/search/{search_term}?searchType={search_type}&ai_model={search_models}&method={method}"

def get_openartai_discovery(driver: WebDriver):
    driver.get("https://openart.ai/discovery")

def openartai_search_prompts(search_term: str, driver: WebDriver):
    search_input = driver.find_element(By.CSS_SELECTOR, 'input[id=":R36ilaqplal6:"]')
    # Click on input field
    search_input.click()
    # Put text in input
    search_input.send_keys(search_term)
    # Simulate pressing the Enter key
    search_input.send_keys(Keys.ENTER)

def check_if_image_exists(images: List[AIImage], image_url: str) -> bool:
    for img in images:
        if img.image_url == image_url:
            return True
    return False

def apply_filters(driver: WebDriver, preiod_wait_in_sec=1):
    # Open model selection filter field
    driver.find_element(By.CLASS_NAME, 'MuiFormControlLabel-root').click()
    time.sleep(preiod_wait_in_sec)
    # deactivate Stable Diffusion
    driver.find_elements(By.CLASS_NAME, 'MuiFormControlLabel-root')[1].click()
    time.sleep(preiod_wait_in_sec)
    # deactivate DALL-E 2
    driver.find_elements(By.CLASS_NAME, 'MuiFormControlLabel-root')[2].click()

def extract_midjourney_images(driver: WebDriver, crawling_progress_bar, progress: int, progress_max=90) -> List[AIImage]:
    try:
        # if we have a presentation view, driver should include only images for this view
        driver_view = driver.find_elements(By.XPATH, "//div[@role='presentation']")[-1]
    except:
        driver_view = driver
    expand_prompt_text(driver)
    # 1. Scroll to bottom in order to load more images
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Note: loading circle has role "progressbar"
    wait_until_element_disappears(driver, '//*[@role="progressbar"]', timeout=10)

    # Make sure after loadingbar disappeared, that image/prompt elements are ready for crawling
    time.sleep(2)
    expand_prompt_text(driver)

    save_html(driver)

    # 2. bring grid elements in right order for later screen scrolling
    columns: List[WebElement] = driver_view.find_elements(By.XPATH, ".//*[contains(@style, 'flex-direction: column')]")
    if len(columns) == 0:
        print("Retry extracting columns")
        columns: List[WebElement] = driver_view.find_elements(By.XPATH, ".//*[contains(@style, 'flex-direction:column')]")

    grid_columns, gridcells = get_girdcells(columns)

    # 3. Extract image and prompt templates
    progress_left = progress_max - progress
    print(f"Found {len(gridcells)} gridcells with potential image and prompt pairs. {len(columns)} columns, {len(grid_columns)} grid_columns")

    expected_num_ai_images = len(gridcells)
    ai_images = extract_image_prompt_pairs(crawling_progress_bar, driver, gridcells, progress, progress_left)

    # Retry extracting ai images
    if len(ai_images) < (expected_num_ai_images/2):
        columns: List[WebElement] = driver_view.find_elements(By.XPATH,
                                                              ".//*[contains(@style, 'flex-direction: column')]")
        if len(columns) == 0:
            print("Retry extracting columns")
            columns: List[WebElement] = driver_view.find_elements(By.XPATH,
                                                                  ".//*[contains(@style, 'flex-direction:column')]")
        grid_columns, gridcells = get_girdcells(columns)
        ai_images = extract_image_prompt_pairs(crawling_progress_bar, driver, gridcells, progress, progress_left)

    return ai_images


def get_girdcells(columns):
    grid_columns: List[List[WebElement]] = []
    for column in columns:
        grid_columns.append(column.find_elements(By.CLASS_NAME, 'MuiCard-root'))
    gridcells = []
    if len(grid_columns) > 0:
        for i in range(len(grid_columns[0])):
            for grid_column in grid_columns:
                with suppress(IndexError):
                    gridcells.append(grid_column[i])
    return grid_columns, gridcells


def extract_image_prompt_pairs(crawling_progress_bar, driver, gridcells, progress, progress_left):
    ai_images: List[AIImage] = []
    for i, gridcell in enumerate(gridcells):
        try:
            # Scroll to the element using JavaScript
            driver.execute_script("arguments[0].scrollIntoView();", gridcell)
            # Wait until image tag is loaded
            image_element = wait_until_image_loaded(gridcell)
            # Extract image webp element
            image_url = image_element.get_attribute('src')
            # catch wrong template image
            if "image_1685064640647_1024" in image_url:
                continue
            assert any(image_url.endswith(ending) for ending in
                       [".webp", ".jpg", "jpeg", ".png"]), f"image_url {image_url}, is not in the expected image format"
            # extract prompt from text area
            ## Note: old way to extract prompt
            # text_elements = gridcell.find_elements(By.CLASS_NAME, "MuiTypography-body2")
            # # Assumption: Last text element is "remix" button, first one is optionally the user name (if existent)
            # text_list = [text_element.text for text_element in text_elements if not has_button_ancestor(text_element)]
            # prompt = text_list[-1]
            ## Note: Prompt is now placed inside of image alt
            prompt = image_element.get_attribute('alt').removeprefix("Prompt: ")

            if not check_if_image_exists(ai_images, image_url):
                ai_images.append(AIImage(image_url=image_url, prompt=prompt))
            crawling_progress_bar.progress(int(progress + (progress_left * (i / len(gridcells)))),
                                           text="Crawling Midjourney images" + ": Crawling...")

        except Exception as e:
            print("Could not extract image and prompt", str(e))
            continue
    return ai_images


def wait_until_image_loaded(gridcell, wait_secs=1):
    # Define the locator for the image element
    image_locator = (By.CSS_SELECTOR, "img[src$='.webp'], img[src$='.jpg'], img[src$='.jpeg'], img[src$='.png']")
    # Wait until the image element is visible
    wait = WebDriverWait(gridcell, wait_secs)
    image_element = wait.until(EC.visibility_of_element_located(image_locator))
    return image_element


def expand_prompt_text(driver):
    # Click on all more to make prompt completly visible
    more_elements = driver.find_elements(By.XPATH, "//span[text()='[more]']")
    for i, more_element in enumerate(more_elements):
        try:
            # Scroll to the element using JavaScript
            #driver.execute_script("arguments[0].scrollIntoView();", more_element)
            #more_element.click()
            driver.execute_script("arguments[0].click();", more_element)
        except Exception as e:
            print(f"more element number {i} is not clickable")
            continue

def click_image(driver, prompt):
    # cut prompt to handle ' and " char
    prompt = prompt[:prompt.find("'")]
    prompt = prompt[:prompt.find('"')]
    #prompt_web_elements = driver.find_elements(By.XPATH, f"//span[text()='{prompt}']")
    #prompt_web_elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{prompt}')]")
    prompt_web_elements = driver.find_elements(By.XPATH, f'//img[contains(@alt, "{prompt}")]')

    if len(prompt_web_elements) == 0:
        st.error("Could not find element. Please try another one.")
        return None
    prompt_web_element = prompt_web_elements[0]
    gridcell = prompt_web_element.find_elements(By.XPATH, "./ancestor::div[contains(@class, 'MuiPaper-root')]")[-1]

    # Scroll to the element using JavaScript
    driver.execute_script("arguments[0].scrollIntoView();", gridcell)
    # Wait until image tag is loaded
    try:
        image_element = wait_until_image_loaded(gridcell)
    except Exception as e:
        logging.warning(e)
        image_element = gridcell.find_element(By.CSS_SELECTOR, "img[src$='.webp'], img[src$='.jpg'], img[src$='.jpeg'], img[src$='.png']")

    # click image
    driver.execute_script("arguments[0].click();", image_element)
    #image_element.click()

def crawl_openartai(crawling_tab):
    set_session_state_if_not_exists()
    progress_text = "Crawling openart ai images"
    crawling_progress_bar = crawling_tab.progress(0, text=progress_text)
    crawling_progress_bar.progress(10,text=progress_text + ": Setup...")
    session_state: SessionState = st.session_state["session_state"]
    driver = session_state.browser.driver
    crawling_progress_bar.progress(20,text=progress_text + ": Search...")
    discovery_url = get_discovery_url(session_state.crawling_request.search_term, session_state.crawling_request.image_ais, only_community=False)
    driver.get(discovery_url)
    time.sleep(3)
    crawling_progress_bar.progress(50,text=progress_text + ": Crawling...")
    session_state.crawling_data = CrawlingData(images=extract_midjourney_images(driver, crawling_progress_bar, 50))
    crawling_progress_bar.empty()

def crawl_openartai_similar_images(crawling_tab, image_nr):
    progress_text = "Crawl openart ai images"
    crawling_progress_bar = crawling_tab.progress(0, text=progress_text)
    # Get session templates
    session_state: SessionState = st.session_state["session_state"]
    driver = session_state.browser.driver
    midjourney_image: AIImage = session_state.crawling_data.images[image_nr]

    # Click on selected image
    try:
        click_image(driver, midjourney_image.prompt)
    except Exception as e:
        st.warning("Could not crawl similar images")
    time.sleep(1)
    crawling_progress_bar.progress(30,text=progress_text + ": Crawling...")

    # Crawl similar images
    session_state.crawling_data = CrawlingData(images=extract_midjourney_images(driver, crawling_progress_bar, 30))
    crawling_progress_bar.empty()


