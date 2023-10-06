from email import message
from flask import Flask, jsonify, request
import os
import cv2 as cv
import time
import base64
import json
import asyncio
import websockets
from websockets.exceptions import ConnectionClosedOK

app = Flask(__name__)
connected_clients = set()
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

async def handle_client(websocket, path):
    references_folder= "/home/ubuntu/smartivity-reference/data/references"
    all_refereces = os.listdir(references_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ref_image_files = [f for f in all_refereces if any(f.lower().endswith(ext) for ext in image_extensions)]
    try:
        input_message = await websocket.recv()
        input_file = base64.b64decode(input_message["input_image"])
        file_name = input_message['input_filename']
        filename = os.path.basename(file_name)

        print("11111111111")

        print(type(input_file))
        print(filename)
        input_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_filename):
            with open(f"{references_folder}/{filename}", "wb") as img_file:
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
    img1 = cv.imread(input_img_path, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(reference_img_path, cv.IMREAD_GRAYSCALE)

    # Ensure the images are not empty
    assert img1 is not None, "could not read image 1"
    assert img2 is not None, "Could not read the image 2."

    # Assuming the images are already loaded as 'img1' and 'img2'

    # Time to detect and describe
    start_time_detect_describe = time.time()

    # Create the CUDA ORB detector
    orb = cv.cuda_ORB.create()

    # Upload images to GPU memory
    gpu_img1 = cv.cuda_GpuMat(img1)
    gpu_img2 = cv.cuda_GpuMat(img2)

    # Detect and compute keypoints and descriptors
    keypoints1_gpu, descriptors1_gpu = orb.detectAndComputeAsync(gpu_img1, None)
    keypoints2_gpu, descriptors2_gpu = orb.detectAndComputeAsync(gpu_img2, None)

    # # Convert keypoints from GPU to CPU memory
    # keypoints1 = orb.convert(keypoints1_gpu)
    # keypoints2 = orb.convert(keypoints2_gpu)

    # Download descriptors from GPU to CPU memory
    descriptors1 = descriptors1_gpu.download()
    descriptors2 = descriptors2_gpu.download()

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

    return (time_taken_match+time_taken_detect_describe, good_matches)


if __name__ == '__main__':
    start_server = websockets.serve(handle_client, "0.0.0.0", 8765, max_size = 3*(10**7),)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()