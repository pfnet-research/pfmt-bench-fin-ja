import os
import time

import anthropic
import openai
from dotenv import load_dotenv
from fastchat.llm_judge.common import API_ERROR_OUTPUT
from fastchat.llm_judge.common import API_MAX_RETRY
from fastchat.llm_judge.common import API_RETRY_SLEEP

ANTHROPIC_MODEL_LIST_NEW = (
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20240620",
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
api_type = os.environ.get("OPENAI_API_TYPE", "openai")
api_version = os.environ.get("OPENAI_API_VERSION")
api_key = os.environ.get("OPENAI_API_KEY")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def chat_completion_openai(model, conv, temperature, max_tokens, **kwargs):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            if api_type == "openai":
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
            elif api_type == "azure":
                client = openai.AzureOpenAI(
                    azure_endpoint=base_url,
                    api_key=api_key,
                    api_version=api_version,
                )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.BadRequestError as e:
            print(type(e), e)
            return ""
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def completion_openai(model, conv, temperature, max_tokens, **kwargs):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            prompt = conv.get_prompt()
            if api_type == "openai":
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
            elif api_type == "azure":
                client = openai.AzureOpenAI(
                    azure_endpoint=base_url,
                    api_key=api_key,
                    api_version=api_version,
                )
            response = client.completions.create(
                model=model,
                prompt=prompt,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].text
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def anthropic_chat_completion_new(model, conv, temperature, max_tokens, **kwargs):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_anthropic_vision_api_messages()[1:]
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            output = response.content[0].text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output
