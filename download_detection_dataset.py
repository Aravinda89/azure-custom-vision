from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials
import requests
import os
import json
from dotenv import load_dotenv

# Load Azure credentials from .env file
load_dotenv()
ENDPOINT = os.getenv("ENDPOINT")
TRAINING_KEY = os.getenv("TRAINING_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# Create a folder for downloaded images and annotations
EXPORT_FOLDER = "detection_data"
ANNOTATIONS_FILE = "annotations.json"
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Authenticate with Azure Custom Vision API
trainer = CustomVisionTrainingClient(ENDPOINT, ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY}))

# Function to fetch all tagged images in batches
def get_all_tagged_images(project_id, max_per_call=256):
    all_images = []
    skip = 0
    while True:
        batch = trainer.get_tagged_images(project_id, take=max_per_call, skip=skip)
        if not batch:
            break
        all_images.extend(batch)
        skip += max_per_call
    return all_images

# Retrieve all tagged images
tagged_images = get_all_tagged_images(PROJECT_ID)
print(f"Retrieved {len(tagged_images)} tagged images.")

annotations = []

# Download images and collect bounding box data
for image in tagged_images:
    try:
        # Download image
        response = requests.get(image.original_image_uri)
        if response.status_code == 200:
            file_path = os.path.join(EXPORT_FOLDER, f"{image.id}.jpg")
            with open(file_path, "wb") as file:
                file.write(response.content)

            # Store bounding box annotations
            image_annotations = {
                "image_id": image.id,
                "image_path": file_path,
                "regions": []
            }
            
            # Bounding box annotations will be saved in the format: (left, top, width, height)
            for region in image.regions:
                image_annotations["regions"].append({
                    "tag_name": region.tag_name,
                    "left": region.left,
                    "top": region.top,
                    "width": region.width,
                    "height": region.height
                })

            annotations.append(image_annotations)

    except Exception as e:
        print(f"Error processing image {image.id}: {e}")
        continue

# Save annotations to a JSON file
with open(os.path.join(EXPORT_FOLDER, ANNOTATIONS_FILE), "w") as f:
    json.dump(annotations, f, indent=4)

print("Download complete. Images and annotations saved successfully.")