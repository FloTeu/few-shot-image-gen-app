import time
import logging
import streamlit as st
import math

from typing import List
from contextlib import suppress
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from utils.session import set_session_state_if_not_exists
from utils.data_classes import SessionState, CrawlingData, MidjourneyImage
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



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

def check_if_image_exists(images: List[MidjourneyImage], image_url: str) -> bool:
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

def extract_midjourney_images(driver: WebDriver, crawling_progress_bar, progress: int, progress_max=90) -> List[MidjourneyImage]:
    midjourney_images: List[MidjourneyImage] = []
    try:
        # if we have a presentation view, driver should include only images for this view
        driver_view = driver.find_elements(By.XPATH, "//div[@role='presentation']")[-1]
    except:
        driver_view = driver
    expand_prompt_text(driver)
    # scroll to botton
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)
    expand_prompt_text(driver)

    # bring grid elements in right order to screen scrolling
    columns = driver_view.find_elements(By.XPATH, ".//*[contains(@style, 'flex-direction: column')]")

    grid_columns = []
    for column in columns:
        grid_columns.append(column.find_elements(By.CLASS_NAME, 'MuiCard-root'))
    gridcells = []
    if len(grid_columns) > 0:
        for i in range(len(grid_columns[0])):
            for grid_column in grid_columns:
                with suppress(IndexError):
                    gridcells.append(grid_column[i])
    progress_left = progress_max - progress
    for i, gridcell in enumerate(gridcells):
        # skip if its not a midjourney image
        # if len(gridcell.find_elements(By.XPATH, "//span[text()='Midjourney']")) == 0:
        #     continue

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
            assert any(image_url.endswith(ending) for ending in [".webp", ".jpg", "jpeg", ".png"]) , f"image_url {image_url}, is not in the expected image format"
            # extract prompt from text area
            prompt = gridcell.find_element(By.CLASS_NAME, "MuiTypography-body2").text
            if not check_if_image_exists(midjourney_images, image_url):
                midjourney_images.append(MidjourneyImage(image_url=image_url, prompt=prompt))
            crawling_progress_bar.progress(int(progress + (progress_left * (i/len(gridcells)))), text="Crawling Midjourney images" + ": Crawling...")

        except Exception as e:
            print("Could not extract image and prompt", str(e))
            continue

    return midjourney_images


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
    # cut prompt to handle ' char
    prompt = prompt[:prompt.find("'")]
    #prompt_web_elements = driver.find_elements(By.XPATH, f"//span[text()='{prompt}']")
    prompt_web_elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{prompt}')]")
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
    progress_text = "Crawling Midjourney images"
    crawling_progress_bar = crawling_tab.progress(0, text=progress_text)
    crawling_progress_bar.progress(10,text=progress_text + ": Setup...")
    session_state: SessionState = st.session_state["session_state"]
    driver = session_state.browser.driver
    time.sleep(1)
    crawling_progress_bar.progress(20,text=progress_text + ": Search...")
    get_openartai_discovery(driver)
    crawling_progress_bar.progress(30,text=progress_text + ": Search...")
    openartai_search_prompts(session_state.crawling_request.search_term, driver)
    time.sleep(1)
    crawling_progress_bar.progress(40,text=progress_text + ": Apply filters...")
    #apply_filters(driver)
    time.sleep(2)
    crawling_progress_bar.progress(50,text=progress_text + ": Crawling...")
    session_state.crawling_data = CrawlingData(midjourney_images=extract_midjourney_images(driver, crawling_progress_bar, 50))
    crawling_progress_bar.empty()

def crawl_openartai_similar_images(crawling_tab, image_nr):
    progress_text = "Crawling Midjourney images"
    crawling_progress_bar = crawling_tab.progress(0, text=progress_text)
    # Get session data
    session_state: SessionState = st.session_state["session_state"]
    driver = session_state.browser.driver
    midjourney_image: MidjourneyImage = session_state.crawling_data.midjourney_images[image_nr]

    # Click on selected image
    click_image(driver, midjourney_image.prompt)
    time.sleep(1)
    crawling_progress_bar.progress(30,text=progress_text + ": Crawling...")

    # Crawl similar images
    session_state.crawling_data = CrawlingData(midjourney_images=extract_midjourney_images(driver, crawling_progress_bar, 30))
    crawling_progress_bar.empty()


