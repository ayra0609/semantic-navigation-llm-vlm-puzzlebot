import ollama
import json
import re

def extract_navigation_intent(user_input: str) -> dict:
    """
    Extract navigation intent from natural language instruction.
    Uses Chain-of-Thought prompting with Mistral via Ollama.
    
    Args:
        user_input: Natural language navigation command (e.g. "Move to the sofa")
    
    Returns:
        dict with 'action' and 'target' keys
    """
    prompt = f"""Extract navigation intent from the instruction below.
Think step by step.

Instruction: "{user_input}"

Step 1: What action is requested? (e.g. navigate, go to, move to)
Step 2: What is the target object or location?
Step 3: Return ONLY a JSON object, nothing else.

Output format: {{"action": "navigate_to", "target": "<object>"}}"""

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    content = response['message']['content']
    
    # Extract JSON from response
    match = re.search(r'\{.*?\}', content, re.DOTALL)
    if match:
        return json.loads(match.group())
    else:
        return {"action": "navigate_to", "target": "unknown", "raw": content}


if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "Move to the sofa",
        "Go to the chair",
        "Navigate to the door",
        "Take me to the table",
    ]
    
    print("=== Intent Extraction Test ===\n")
    for cmd in test_inputs:
        result = extract_navigation_intent(cmd)
        print(f"Input:  {cmd}")
        print(f"Output: {result}")
        print()