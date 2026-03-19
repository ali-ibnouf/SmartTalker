import os
from dotenv import load_dotenv
import urllib.request
import json

load_dotenv('.env.production')
api_key = os.environ.get("DASHSCOPE_API_KEY")

url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

req = urllib.request.Request(url, headers=headers)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print("AVAILABLE MODELS:")
        for model in result.get("data", []):
            print(f"- {model.get('id')}")
except Exception as e:
    print(f"Error checking models: {e}")
