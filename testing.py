import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# image=Image.open("download_wikipedia_squash.jpg")
# # Check for cats and remote controls
# text = "a black rubber squash ball."

# inputs = processor(images=image, text=text, return_tensors="pt").to(device)
# with torch.no_grad():
#     outputs = model(**inputs)

# results = processor.post_process_grounded_object_detection(
#     outputs,
#     inputs.input_ids,
#     box_threshold=0.4,
#     text_threshold=0.3,
#     target_sizes=[image.size[::-1]]
# )
# print(results)
from tqdm import tqdm
import os
import cv2
def get_annotations(image_folder, output_folder, text="a small black rubber squash ball."):
    for filename in tqdm(os.listdir(image_folder)):
        image=Image.open(image_folder+'/'+filename)
        inputs=processor(images=image, text=text, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs=model(**inputs)
        results=processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3, 
            target_sizes=[image.size[::-1]]
        )
        print(f'results: {results}')
        annotated_image=image
        print(f'len(results): {len(results)}')
        print(f'len(results[0]): {len(results[0])}')
        print(f'len(results[\'boxes\']): {len(results[0]['boxes'])}')
        print(f'results[0][\'boxes\'][0]: {results[0]['boxes'][0]}')
        print(f'result[0][boxes][0][0]: {results[0]['boxes'][0][0]}')
        cv2.rectangle(annotated_image, (int(results[0]['boxes'][0][0]), int(results[0]['boxes'][0][1])), (int(results[0]['boxes'][0][2]), int(results[0]['boxes'][0][3])), (0,255,0), 5)
        cv2.imshow(image)
        
get_annotations('testing_folder', 'output_folder')