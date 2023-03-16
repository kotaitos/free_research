import os
from os.path import join, dirname
from dotenv import load_dotenv
import openai

load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# APIキーの設定
openai.api_key = os.environ.get("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "大谷翔平について教えて"},
    ],
)
print(response.choices[0]["message"]["content"].strip())