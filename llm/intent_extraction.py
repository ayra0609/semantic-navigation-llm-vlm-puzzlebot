"""
intent_extraction.py  —  LLM-based navigation intent extractor

Primary:  Ollama (Mistral) via Chain-of-Thought prompting
Fallback: regex-based extraction when Ollama is unavailable
"""

import re
import json

# ── Regex fallback ────────────────────────────────────────────────────────────
_PATTERNS = [
    r'(?:navigate to|go to|move to|head to|drive to|find|reach|approach)\s+(?:the\s+)?(.+)',
    r'(?:take me to|bring me to)\s+(?:the\s+)?(.+)',
    r'(?:target|object)\s*(?:is\s+)?(?:the\s+)?(.+)',
]

def _regex_extract(text: str) -> dict | None:
    for pat in _PATTERNS:
        m = re.search(pat, text.lower().strip())
        if m:
            target = m.group(1).strip().rstrip('.')
            return {'action': 'navigate_to', 'target': target}
    return None


# ── Ollama / Mistral (primary) ────────────────────────────────────────────────
def _ollama_extract(user_input: str) -> dict | None:
    import ollama
    prompt = f"""Extract navigation intent from the instruction below.
Think step by step.

Instruction: "{user_input}"

Step 1: What action is requested? (navigate, go to, move to, etc.)
Step 2: What is the target object or location?
Step 3: Return ONLY a JSON object, nothing else.

Output format: {{"action": "navigate_to", "target": "<object>"}}"""

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    content = response['message']['content']
    print(f'[LLM-Ollama] raw response: {content.strip()}')
    m = re.search(r'\{.*?\}', content, re.DOTALL)
    if m:
        parsed = json.loads(m.group())
        print(f'[LLM-Ollama] parsed: {parsed}')
        return parsed
    print('[LLM-Ollama] WARNING: no JSON found in response')
    return None


# ── Public API ────────────────────────────────────────────────────────────────
def extract_navigation_intent(user_input: str) -> dict:
    """
    Extract navigation intent.  Tries Ollama first; falls back to regex.
    Always returns a dict with at least {'action': ..., 'target': ...}.
    """
    print(f'[LLM] input: "{user_input}"')

    # Try Ollama
    try:
        result = _ollama_extract(user_input)
        if result:
            print(f'[LLM] method=Ollama  result={result}')
            return result
    except Exception as e:
        print(f'[LLM] Ollama unavailable ({type(e).__name__}: {e}) — using regex fallback')

    # Regex fallback
    result = _regex_extract(user_input)
    if result:
        print(f'[LLM] method=regex  result={result}')
        return result

    print(f'[LLM] ERROR: no intent found for "{user_input}"')
    return {'action': 'unknown', 'target': 'unknown', 'raw': user_input}


if __name__ == '__main__':
    tests = [
        'Navigate to the red box',
        'Go to the ArUco marker',
        'Move forward 0.5 metres',
        'Find the blue chair',
    ]
    for t in tests:
        print(f'{t!r:45s} → {extract_navigation_intent(t)}')
