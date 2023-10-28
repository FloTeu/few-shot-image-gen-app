from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By


def has_button_ancestor(element: WebElement) -> bool:
    """Return true if a ancestor is a button"""
    # The xpath to find the closest ancestor button for the element
    xpath = "./ancestor::button"

    # Try to find the ancestor button
    button_ancestors = element.find_elements(By.XPATH, xpath)

    # Return True if any button ancestor is found, otherwise False
    return len(button_ancestors) > 0
