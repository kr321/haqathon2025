from detector import detect_objects
from chatbot import get_response
from PIL import Image

if __name__ == '__main__':
    image_path = 'sample_images/scene.jpg'

    # Step 1: Run object detection
    detected_objects = detect_objects(image_path)
    print("Detected:", detected_objects)

    # Step 2: Query chatbot (LLM-based) for advice
    response = get_response(detected_objects)
    print("\nResponse:\n", response)

