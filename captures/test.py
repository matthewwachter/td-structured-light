import cv2
import glob
import traceback
import os
from pprint import pprint
import numpy as np

import re

# Set the correct width and height that matches your setup
width, height = 1280, 720  # Example dimensions, replace with your actual dimensions


def sort_key_func(filename):
    # This regular expression extracts the number from the filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Convert the first found number in the filename to an integer
        return int(numbers[0])
    return 0  # Default value if no number is found

# Replace 'path/to/pattern_images' with the path to your folder containing the captured Gray Code pattern images
print("Current Working Directory:", os.getcwd())
pattern_folder = './captures'

# Loading pattern images
pattern_images = glob.glob('./captures/capture_*.tiff')
#print(pattern_images)

# Sort the file paths to ensure they are in the correct order
pattern_images.sort(key=sort_key_func)
#pprint(pattern_images)

# Load images into OpenCV
pattern_images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in pattern_images]

# Load the black and white images
black_img = cv2.imread('./captures/black.tiff', cv2.IMREAD_GRAYSCALE)
white_img = cv2.imread('./captures/white.tiff', cv2.IMREAD_GRAYSCALE)

white_images = [np.ones((height, width), dtype=np.float32), white_img]
black_images = [np.zeros((height, width), dtype=np.float32), black_img]



# Create a GrayCodePattern object


# Prepare an empty Mat for the disparity map
disparityMap = np.zeros((height, width), dtype=np.float64)

grey_code = cv2.structured_light_GrayCodePattern.create(width, height)

success, patterns = grey_code.generate()

cam_images = [patterns, pattern_images]

print(width, height)
# Decoding
try:
    success, disparityMap = grey_code.decode(cam_images, disparityMap, black_images, white_images, flags=cv2.structured_light.DECODE_3D_UNDERWORLD)
    
    if success:
        print(disparityMap)
        # Normalize the disparity map to 0-255 range
        disparityMap_normalized = cv2.normalize(disparityMap, None, alpha=0, beta=255, 
                                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Replicate the disparity map across R, G, and B channels
        rgb_disparityMap = cv2.cvtColor(disparityMap_normalized, cv2.COLOR_GRAY2RGB)

        # Create an alpha channel with constant value 255
        alpha_channel = np.ones(disparityMap_normalized.shape, dtype=disparityMap_normalized.dtype) * 255

        # Merge to form an RGBA image
        rgba_disparityMap = cv2.merge((rgb_disparityMap, alpha_channel))

        # Save the RGBA image as TIFF
        cv2.imwrite('decoded_image1.tiff', disparityMap_normalized)
        # Process your decoded image here
        #cv2.imwrite('decoded_image.tiff', disparityMap)

        cv2.imshow("disparityMap", disparityMap_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('success:', success)
        pass
    else:
        print("Decoding failed")

except Exception as e:
    print(f"Decoding error: {e}")
    print("Number of images:", len(pattern_images))
    #pprint([img.shape for img in pattern_images])
    traceback.print_exc()
    # Add additional diagnostics if necessary


# Check if there are any images to display
    if pattern_images:
        # Scale factor for resizing images (e.g., 0.25 will reduce each dimension by 75%)
        scale_factor = 0.25

        # Determine the size of the grid
        grid_size = int(np.ceil(np.sqrt(len(pattern_images))))
        
        # Assuming all images are of the same size, get the dimensions of the first image
        img_height, img_width = pattern_images[0][0].shape[:2]
        
        # Calculate new dimensions
        new_height, new_width = int(img_height * scale_factor), int(img_width * scale_factor)

        # Create a 3-channel BGR canvas for the montage
        montage = np.zeros((new_height * grid_size, new_width * grid_size, 3), dtype=np.uint8)

        # Font settings for the index number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .75
        font_color = (0, 0, 255)  # White color
        thickness = 2
        line_type = cv2.LINE_AA

        # Place each resized image in the montage and then add index number in red
        for i, img in enumerate(pattern_images):
            if img is not None:
                resized_img = cv2.resize(img, (new_width, new_height))
                bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR
                x = (i % grid_size) * new_width
                y = (i // grid_size) * new_height

                montage[y:y+new_height, x:x+new_width] = bgr_img

                # Put the index number on the montage in red
                text_pos = (x + new_width - 50, y + 30)  # Adjust position as needed
                cv2.putText(montage, str(i), text_pos, font, font_scale, font_color, thickness, line_type)



        # Display the montage
        cv2.imshow("Image Montage", montage)
        #cv2.imshow(pattern_images[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No images available to display.")