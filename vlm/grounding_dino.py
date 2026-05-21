from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_processor = None
_model = None
_device = None

def load_model():
    """Load Grounding DINO model (cached after first call)."""
    global _processor, _model, _device
    if _model is None:
        print("Loading Grounding DINO...")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(_device)
        print(f"Model ready on {_device}")
    return _processor, _model, _device


def detect_object(image: Image.Image, target: str, threshold: float = 0.25) -> dict:
    """
    Detect target object in image using Grounding DINO.

    Args:
        image: PIL Image
        target: object name (e.g. "sofa")
        threshold: detection confidence threshold

    Returns:
        dict with 'found', 'box', 'score', 'center_x', 'center_y', 'image_width', 'image_height'
    """
    processor, model, device = load_model()

    text = f"{target}."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=threshold, text_threshold=threshold,
            target_sizes=[image.size[::-1]])
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=threshold, text_threshold=threshold,
            target_sizes=[image.size[::-1]])

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]

    if len(scores) == 0:
        return {"found": False, "target": target}

    # Take highest confidence detection
    best_idx = scores.argmax()
    box = boxes[best_idx].tolist()
    score = scores[best_idx].item()

    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return {
        "found": True,
        "target": target,
        "score": round(score, 3),
        "box": [round(v, 1) for v in box],       # [x1, y1, x2, y2]
        "center_x": round(center_x, 1),
        "center_y": round(center_y, 1),
        "image_width": image.size[0],
        "image_height": image.size[1],
    }


if __name__ == "__main__":
    # Test with COCO sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    for target in ["sofa", "cat", "table"]:
        result = detect_object(image, target)
        print(result)