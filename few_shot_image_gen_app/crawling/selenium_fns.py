from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def wait_until_element_disappears(driver: webdriver, web_element_xpath: str, timeout=10):
    try:
        element_present = EC.presence_of_element_located((By.XPATH, web_element_xpath))
        WebDriverWait(driver, timeout).until_not(element_present)
    except TimeoutException:
        print("Timed out waiting for the progress bar to disappear")

def has_button_ancestor(element: WebElement) -> bool:
    """Return true if a ancestor is a button"""
    # The xpath to find the closest ancestor button for the element
    xpath = "./ancestor::button"

    # Try to find the ancestor button
    button_ancestors = element.find_elements(By.XPATH, xpath)

    # Return True if any button ancestor is found, otherwise False
    return len(button_ancestors) > 0

def save_html(driver: webdriver):
    html = driver.page_source
    with open("test.html", "w") as fp:
        fp.write(html)