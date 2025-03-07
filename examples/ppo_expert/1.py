import os
import base64
import requests
from tkinter import Tk, Label, Button, filedialog, Frame
from PIL import Image, ImageTk

# 设置OpenAI的API密钥
api_key = 'sk-proj-8DZ8o0Fes75agebcxrCKT3BlbkFJBwM8ITd2R1PkJiCB7Hnk'


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Function to send image to OpenAI and get the response
def get_image_classification(image_path):
    base64_image = "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"


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = "Tell me how many lane lines there are."

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


classification = get_image_classification("11")
print(classification)