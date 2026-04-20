import os

MODEL_NAME_deepseek = "deepseek-chat"
LITELLM_MODEL_NAME_deepseek = os.getenv(
    "LITELLM_MODEL_NAME_deepseek",
    f"deepseek/{MODEL_NAME_deepseek}",
)
API_BASE_deepseek = "https://api.deepseek.com"
API_KEY_deepseek = os.getenv("DEEPSEEK_API_KEY", "")
CHAT_EP_deepseek = API_BASE_deepseek.rstrip("/") + "/chat/completions"
COMP_EP_deepseek = API_BASE_deepseek.rstrip("/") + "/completions"



def get_deepseek_api_key(api_key=None):
    return (api_key or API_KEY_deepseek or "").strip()


def get_deepseek_headers(api_key=None):
    headers = {"Content-Type": "application/json"}
    resolved_api_key = get_deepseek_api_key(api_key)
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    return headers
