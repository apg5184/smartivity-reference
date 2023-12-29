import asyncio
import base64
import json
import os
import pickle
import time
from email import message

import cv2 as cv
import websockets
from flask import Flask, jsonify, request
from websockets.exceptions import ConnectionClosedOK

print("yesssssssssssss!!!!!!!!!!!!")
print(cv.cuda.getCudaEnabledDeviceCount())
cv.cuda.setDevice(0)

app = Flask(__name__)
connected_clients = set()
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
root_dir = "/home/ubuntu/smartivity-reference"
descriptor_file = os.path.join(root_dir, "descriptors.pkl")

if not os.path.exists(descriptor_file):
    with open(descriptor_file, 'wb') as f:
        pass

init_len = 0

with open(descriptor_file, 'rb') as f:
    try:
        desc_data = pickle.load(f)
        init_len = len(desc_data)
    except EOFError:
        desc_data = {}

heights_file = os.path.join(root_dir, "heights.pkl")
widths_file = os.path.join(root_dir, "widths.pkl")
if not os.path.exists(heights_file):
    with open(heights_file, 'wb') as f:
        pickle.dump({}, f)
if not os.path.exists(widths_file):
    with open(widths_file, 'wb') as f:
        pickle.dump({}, f)
with open(heights_file, 'rb') as f:
    heights_data = pickle.load(f)
with open(widths_file, 'rb') as f:
    widths_data = pickle.load(f)

async def handle_client(websocket, path):
    references_folder= os.path.join(root_dir, "data/references")
    all_refereces = os.listdir(references_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ref_image_files = [f for f in all_refereces if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    try:
        input_message = await websocket.recv()
        input_message = json.loads(input_message)
        input_file = base64.b64decode(input_message["input_image"])
        file_name = input_message['input_filename']
        filename = os.path.basename(file_name)

        print("11111111111")

        print(type(input_file))
        print(filename)
        input_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_filename):
            with open(input_filename, "wb") as img_file:
                img_file.write(input_file)
                print("Received and saved image from client.")

        print("2222222222")

        total_time = 0.0
        max_matches = 0
        best_match = "no match"

        for idx, image_file in enumerate(ref_image_files):
            ref_filename = os.path.join(references_folder, image_file)
            time_taken, good_matches = cuda_orb_match(input_filename, ref_filename)

            total_time += time_taken

            if max_matches < len(good_matches):
                max_matches = len(good_matches)
                best_match = ref_filename


        print("33333333")
        message={
            "best match": best_match
        }

        print("Doneeeee      Timeeeeeeee: ", total_time)
        await websocket.send(json.dumps(message))
    except ConnectionClosedOK:
        print("WebSocket connection closed gracefully.")
    finally:
        connected_clients.remove(websocket)

def cuda_orb_match(input_img_path, reference_img_path):
    # return (0,[])
    print(input_img_path)
    print(reference_img_path)

    ref_image_basename = os.path.basename(reference_img_path)
    input_image_basename = os.path.basename(input_img_path)

    # Time to detect and describe
    start_time_detect_describe = time.time()

    
    if input_image_basename not in desc_data:
        img1 = cv.imread(input_img_path, cv.IMREAD_GRAYSCALE)

        # Ensure the images are not empty
        assert img1 is not None, "could not read image 1"

        # Create the CUDA ORB detector
        orb = cv.cuda_ORB.create()

        # Upload images to GPU memory
        gpu_img1 = cv.cuda_GpuMat(img1)
    
        # Detect and compute keypoints and descriptors
        keypoints1_gpu, descriptors1_gpu = orb.detectAndComputeAsync(gpu_img1, None)

        # Download descriptors from GPU to CPU memory
        descriptors1 = descriptors1_gpu.download()
    
        desc_data[input_image_basename] = descriptors1
    else:
        descriptors1 = desc_data[input_image_basename]
    

    if ref_image_basename not in desc_data:
        img2 = cv.imread(reference_img_path, cv.IMREAD_GRAYSCALE)

        # Ensure the images are not empty
        assert img2 is not None, "Could not read the image 2."

        # Assuming the images are already loaded as 'img1' and 'img2'

        # Create the CUDA ORB detector
        orb = cv.cuda_ORB.create()

        # Upload images to GPU memory
        gpu_img2 = cv.cuda_GpuMat(img2)

        # Detect and compute keypoints and descriptors
        keypoints2_gpu, descriptors2_gpu = orb.detectAndComputeAsync(gpu_img2, None)

        # Download descriptors from GPU to CPU memory
        descriptors2 = descriptors2_gpu.download()

        desc_data[ref_image_basename] = descriptors2
    else:
        descriptors2 = desc_data[ref_image_basename]

    if init_len != len(desc_data):
        with open(descriptor_file, 'wb') as f:
            pickle.dump(desc_data, f)

    end_time_detect_describe = time.time()

    time_taken_detect_describe = end_time_detect_describe - start_time_detect_describe

    # print("time taken describe", time_taken_detect_describe)

    # Time to match
    start_time_match = time.time()

    # Matching descriptors using BFMatcher (Brute Force Matcher)
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)


    end_time_match = time.time()

    time_taken_match = end_time_match - start_time_match

    return (time_taken_match + time_taken_detect_describe, good_matches)


if __name__ == '__main__':
    start_server = websockets.serve(handle_client, "0.0.0.0", 8765, max_size = 3*(10**7),)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()