import cv2
import numpy as np
import os
import sys

def cylindrical_projection(image, K):
    """Applies inverse cylindrical projection to correct distortions."""
    h, w = image.shape[:2]
    f = K[0, 0]  # Focal length
    
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)
    
    xc, yc = w // 2, h // 2  # Image center

    for y in range(h):
        for x in range(w):
            theta = np.arctan((x - xc) / f)
            h_new = (y - yc) / f
            X = np.sin(theta)
            Z = np.cos(theta)

            x_map[y, x] = f * X + xc
            y_map[y, x] = f * h_new + yc

    cylindrical_img = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cylindrical_img

def detect_and_match_features(img1, img2):
    """Detect and match features using SIFT with RANSAC filtering."""
    sift = cv2.SIFT_create()
    
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    keypoints1, descriptors1 = sift.detectAndCompute(gray_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_img2, None)
    
    if descriptors1 is None or descriptors2 is None:
        print("DEBUG: Not enough keypoints detected.")
        return None, None, None
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good_matches) < 4:
        print("DEBUG: Not enough good matches found.")
        return None, None, None
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, keypoints1, keypoints2

def stitch_images(image_files, output_filename):
    """Stitches images together using computed homographies. Supports all camera movement directions."""
    images = [cv2.imread(f) for f in sorted(image_files)]
    base_image = images[0]
    result = base_image.copy()
    
    for i in range(1, len(images)):
        H, _, _ = detect_and_match_features(result, images[i])
        
        if H is None:
            print(f"Skipping image {i} due to insufficient matches.")
            continue
        
        result = cv2.warpPerspective(result, H, (result.shape[1] + images[i].shape[1], result.shape[0] + images[i].shape[0]))
        result[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]
    
    cv2.imwrite(output_filename, result)
    print(f"Panorama saved as {output_filename}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python prog1.py <input_image_directory> <output_panorama_image>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.JPG'))]
    if not image_files:
        print("No images found in the directory.")
        sys.exit(1)
    
    stitch_images(image_files, output_filename)
    
if __name__ == "__main__":
    main()
