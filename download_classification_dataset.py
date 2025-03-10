from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from msrest.authentication import ApiKeyCredentials
import requests
import os
import csv
from dotenv import load_dotenv

# Load Azure credentials from .env file
load_dotenv()

ENDPOINT = os.getenv("ENDPOINT")
TRAINING_KEY = os.getenv("TRAINING_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# Create a directory to save images
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# CSV file for storing labels
CSV_FILE = os.path.join(SAVE_DIR, "tags.csv")

# Retrieving images in batches 
def get_tagged_images(project_id, batch_size=256):
    """
    Fetch all tagged images from the Azure Custom Vision project.
    """
    images, skip = [], 0
    while True:
        try:
            batch = trainer.get_tagged_images(project_id, take=batch_size, skip=skip)
            if not batch:
                break
            images.extend(batch)
            skip += batch_size
        except Exception as e:
            print(f"Error retrieving images: {e}")
            break
    return images

# Authenticate with Azure Custom Vision
trainer = CustomVisionTrainingClient(ENDPOINT, ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY}))

# Retrieve tagged images
tagged_images = get_tagged_images(PROJECT_ID)
print(f"Retrieved {len(tagged_images)} tagged images.")

# Download images and save tags
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image ID", "Tags"])

    for image in tagged_images:
        img_path = os.path.join(SAVE_DIR, f"{image.id}.jpg")

        try:
            # Download and save image
            response = requests.get(image.original_image_uri, timeout=10)  # Set timeout to 10 sec
            response.raise_for_status()  # Raise an error for bad HTTP responses (e.g., 404, 500)
            
            with open(img_path, "wb") as img_file:
                img_file.write(response.content)

            # Save image tags
            writer.writerow([image.id, ",".join(tag.tag_name for tag in image.tags)])

        except requests.exceptions.RequestException as e:
            print(f"Skipping image {image.id} due to download error: {e}")
        except Exception as e:
            print(f"Skipping image {image.id} due to unexpected error: {e}")

print("Images and tags saved successfully.")