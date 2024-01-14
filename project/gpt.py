import openai as gpt
import os

def get_gpt_response(messages):
    gpt.api_key=os.getenv("OPENAI_API_KEY")
    response = gpt.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        max_tokens=600,
        n=1,
        temperature=0.8,
        stream=True,
    )
    return response
