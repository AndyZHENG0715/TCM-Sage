import os
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('DASHSCOPE_API_KEY')
print(f"API Key found: {bool(api_key)}")

try:
    llm = ChatTongyi(
        model='qwen-plus',
        api_key=api_key
    )
    print("Successfully initialized with api_key")
    # Try a simple invoke to see if it fails
    # res = llm.invoke("Hello")
    # print(res)
except Exception as e:
    print(f"Failed with api_key: {e}")

try:
    llm = ChatTongyi(
        model='qwen-plus',
        dashscope_api_key=api_key
    )
    print("Successfully initialized with dashscope_api_key")
except Exception as e:
    print(f"Failed with dashscope_api_key: {e}")
