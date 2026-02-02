import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

def llm_post_edit(text: str, lang: str) -> str:
    if lang == "ta":
        instruction = (
            "You are a Tamil grammar corrector.\n"
            "TASK:\n"
            "- Fix spelling and grammar ONLY\n"
            "- Do NOT translate\n"
            "- Do NOT explain\n"
            "- Do NOT add English\n"
            "- Output ONLY corrected Tamil\n"
            "- Output ONE sentence only\n"
        )
    elif lang == "hi":
        instruction = (
            "You are a Hindi grammar corrector.\n"
            "TASK:\n"
            "- Fix spelling and grammar ONLY\n"
            "- Do NOT translate\n"
            "- Do NOT explain\n"
            "- Do NOT add English\n"
            "- Output ONLY corrected Hindi\n"
            "- Output ONE sentence only\n"
        )
    else:
        return text

    prompt = f"""
{instruction}

INPUT:
{text}

OUTPUT:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 80,
            "stop": ["\n", "Explanation", "I'm", "I am"]
        }
    }

    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30
    )

    response.raise_for_status()
    output = response.json()["response"].strip()


    output = re.sub(r"[A-Za-z].*", "", output).strip()

    return output if output else text
