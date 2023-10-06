from flask import Flask, jsonify, request
import os
import cv2 as cv
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

print("yesssssssssss")
print(cv.cuda.getCudaEnabledDeviceCount())
cv.cuda.setDevice(0)


@app.route('/', methods=['GET', 'POST'])
def index():
    print("yesssssssssss1111111")
    print(cv.cuda.getCudaEnabledDeviceCount())
    return "Server works!!!"

@app.route('/process', methods=['GET', 'POST'])
def helper():
    print("inside server")
    references_folder= "/home/ubuntu/smartivity-reference/data/references"
    all_refereces = os.listdir(references_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ref_image_files = [f for f in all_refereces if any(f.lower().endswith(ext) for ext in image_extensions)]


    st_time = time.time()
    input_file = request.files['input_image']
    # ref_file = request.files['ref_image']
    print("time to download00000000000: ", time.time()-st_time)

    print("11111111111")

    print(type(input_file))
    print(input_file.filename)
    input_filename = os.path.join(app.config['UPLOAD_FOLDER'], input_file.filename)
    if not os.path.exists(input_filename):
        input_file.save(input_filename)

    print("2222222222")

    # print(type(ref_file))
    # print(ref_file.filename)
    # ref_filename = os.path.join(app.config['UPLOAD_FOLDER'], ref_file.filename)
    # ref_file.save(ref_filename)

    print("3333333333")

    total_time = 0.0
    max_matches = 0
    best_match = "no reference"

    for idx, image_file in enumerate(ref_image_files):
        print(str(idx) + "\n")
        ref_filename = os.path.join(references_folder, image_file)
        time_taken, good_matches = cuda_orb_match(input_filename, ref_filename)

        total_time += time_taken

        if max_matches < len(good_matches):
            max_matches = len(good_matches)
            best_match = ref_filename


    print("4444444444")

    print("Timeeeeeeee: ", total_time)
    print("Matchessssss: ", len(good_matches))
    print("doneeeeee")
    return jsonify(best_match)
    # return send_from_directory(app.config['UPLOAD_FOLDER'], file.filename)



def cuda_orb_match(input_img_path, reference_img_path):
    print("5555555555")
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

    print("time taken describe", time_taken_detect_describe)

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

    print(time_taken_detect_describe, time_taken_match)

    # Draw the matches
    # img_matches = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)

    # cv.imwrite("matches_img.jpg",img_matches)

    return (time_taken_match+time_taken_detect_describe, good_matches)





# def process_image(filepath):
#     with Image.open(filepath) as img:
#         grayscale = img.convert('L')
#         grayscale.save(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
