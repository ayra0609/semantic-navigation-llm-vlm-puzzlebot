from llm.intent_extraction import extract_navigation_intent
from vlm.grounding_dino import detect_object
from PIL import Image
import requests

def run_pipeline(user_input: str, image: Image.Image) -> dict:
    """
    Full pipeline: natural language → object detection
    
    Args:
        user_input: e.g. "Move to the sofa"
        image: PIL Image from robot camera
    
    Returns:
        dict with intent + detection result
    """
    # Step 1: LLM extracts intent
    print(f"\n[LLM] Input: '{user_input}'")
    intent = extract_navigation_intent(user_input)
    print(f"[LLM] Intent: {intent}")

    target = intent.get("target", "unknown")

    if target == "unknown":
        return {"status": "failed", "reason": "Could not extract target"}

    # Step 2: VLM detects target in image
    print(f"[VLM] Detecting '{target}' in image...")
    detection = detect_object(image, target)
    print(f"[VLM] Detection: {detection}")

    return {
        "status": "success",
        "intent": intent,
        "detection": detection
    }


if __name__ == "__main__":
    # Test with sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    test_commands = [
        "Move to the sofa",
        "Go to the cat",
        "Navigate to the table",
    ]

    print("=== Semantic Navigation Pipeline ===")
    for cmd in test_commands:
        result = run_pipeline(cmd, image)
        print(f"Final result: {result}\n")
        print("-" * 50)
