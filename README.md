# midjourney-prompt-generator
[Streamlit application](https://image-gen-ai-app.streamlit.app/) for a Image GenAI prompt generator demo. 

Basic idea: Using a GPT model (atm. ChatGPT) via langchain library and let it learn with a few shot learning technique how Midjourney prompts can be generated.

## Tech Stack
* Crawling: Selenium
* Frontend: Streamlit
* Prompt Engineering/Few Shot Learning: LangChain

[Source code llm-few-shot-generator](https://github.com/FloTeu/llm-few-shot-generator)
![alt text](assets/tech_stack.jpg "Teck Stack")



## Example
Text Prompt Input: "Grandma" 

Midjourney Prompt Generator output images:
![alt text](assets/grandmas.jpg "Grandmas")




## Local Setup

### Prerequisites
Create a [streamlit secret](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management) TOML file before running this app locally.
```console
mkdir .streamlit
printf 'open_ai_api_key = ""' > .streamlit/secrets.toml
```
If it not exists yet, you have to create a open ai api key:
1. openai api key: https://platform.openai.com/account/api-keys

Setup Dependency Environment with poetry 
Install poetry on your device (https://python-poetry.org/docs/)
```shell
# install all dependencies
poetry install
# Enter shell within poetry environment
poetry shell
```

### Run
Optionally set environment variable DEBUG = 1, in order to run selenium without headless mode
```console
streamlit run app.py
```

### TODOs
- [ ] Include LLM prompt with highlited prompt elements
- [x] Ensure crawled data is also display
- [ ] Update Prompt Engineering class for SDXL