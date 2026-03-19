import os
import sys
import json
import urllib.request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.environ.get("LLM_API_KEY")
if not api_key:
    # Try getting it from DASHSCOPE_API_KEY
    api_key = os.environ.get("DASHSCOPE_API_KEY")

if not api_key:
    print("ERROR: No DashScope API key found in LLM_API_KEY or DASHSCOPE_API_KEY.")
    print("Available OS env vars:", list(os.environ.keys()))
    sys.exit(1)

print(f"Using API Key: {api_key[:4]}...{api_key[-4:]}")

# Test DashScope API using compatible OpenAI format
url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen3-max",
    "messages": [
        {"role": "user", "content": "Hello, this is a test. Please reply 'Test successful'."}
    ]
}

req = urllib.request.Request(url, headers=headers, data=json.dumps(data).encode("utf-8"))

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print("SUCCESS! API responded:")
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} {e.reason}")
    print(e.read().decode())
except Exception as e:
    print(f"Error: {e}")
