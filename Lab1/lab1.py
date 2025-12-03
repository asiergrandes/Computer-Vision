import cv2
from matplotlib import pyplot as plt
import numpy as np


##############
## TASK 1.1 ##
##############

# a) Read images and extract pixel values

# Load 'lena.tiff' and convert from BGR (OpenCV default) to RGB
im = cv2.imread('lena.tiff')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
print(f'{im.shape=}')
print(f'Color of pixel (350,100) in lena.tiff: {im[350,100,:]}')

# Load 'baboon.tiff' and convert to RGB
im = cv2.imread('baboon.tiff')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
print(f'{im.shape=}')
print(f'Color of pixel (350,100) in baboon.tiff: {im[350,100,:]}')

# Open and read frames from a video
cap = cv2.VideoCapture('Holywood2-t00427-rgb.avi')
cont = 0

if not cap.isOpened():
    print('Error opening video')
    exit()

while True:
    # Capture the next frame
    ret, frame = cap.read()
    cont += 1

    # Stop if no more frames
    if not ret:
        print('End of video or frame read error.')
        break

    # Print RGB pixel values for frames 50 to 60
    if 49 < cont < 61:
        rgb_pixel = frame[200, 100, ::-1]  # reverse channels BGR → RGB
        print(f'Frame {cont} – pixel (200,100):', rgb_pixel)

cap.release()

# b) Invert colors of a specific region in the image

im = cv2.imread('lena.tiff')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Define region
x_start, x_end = 200, 250
y_start, y_end = 200, 250

# Extract and invert that region
cut = im[x_start:x_end, y_start:y_end]
inv = 255 - cut
im[x_start:x_end, y_start:y_end] = inv

# Display result
plt.imshow(im)
plt.title("lena.tiff inverted")
plt.show()

# Save inverted image
im_final = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
cv2.imwrite('lena_inverted.tiff', im_final)


##############
## TASK 1.2 ##
##############

# a) Manual grayscale conversion

im = cv2.imread('lena.tiff')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Convert to grayscale using luminance formula
im_gray = im[:, :, 0]*0.299 + im[:, :, 1]*0.587 + im[:, :, 2]*0.114
# Use cmap because matplotlib thinks it is in RGB, while it is grayscale
plt.imshow(im_gray, cmap='gray')
plt.show()

# Save as uint8 grayscale
cv2.imwrite('lena_gray.tiff', im_gray.astype(np.uint8))

# Repeat for baboon
im = cv2.imread('baboon.tiff')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_gray = im[:, :, 0]*0.299 + im[:, :, 1]*0.587 + im[:, :, 2]*0.114
plt.imshow(im_gray, cmap='gray')
plt.show()

cv2.imwrite('baboon_gray.tiff', im_gray.astype(np.uint8))

# Question: Why is simply averaging the color channels not a good approach for computing grayscale images?
# Answer: If we just mix the red, green, and blue colors equally, the picture won’t look right in black and white,
# because our eyes see green as brighter and blue as darker. The special formula fixes this so the gray picture looks more like what we really see.


# b) Recolor a grayscale image based on intensity thresholds

def recolor(im_gray, low, high, colors):
    """
    Recolor a grayscale image by assigning different colors
    to three intensity ranges: low, mid, and high.
    """
    color_image = np.zeros((im_gray.shape[0], im_gray.shape[1], 3), dtype=np.uint8)

    # Create threshold masks
    mask_low = im_gray < low
    mask_mid = (im_gray >= low) & (im_gray <= high)
    mask_high = im_gray > high

    # Apply RGB colors to each range
    color_image[mask_low] = colors[0]
    color_image[mask_mid] = colors[1]
    color_image[mask_high] = colors[2]

    return color_image


im_gray = cv2.imread('baboon_gray.tiff', cv2.IMREAD_GRAYSCALE)
im_recolor = recolor(im_gray, low=100, high=200, colors=[(255,0,0), (0,255,0), (0,0,255)])  # RGB
plt.imshow(im_recolor)
plt.title("Recolored image")
plt.show()
cv2.imwrite('recolored_image.tiff', cv2.cvtColor(im_recolor, cv2.COLOR_RGB2BGR))


# c) Convert a rectangular region of an image to grayscale

def partial_grayscale(color_image, top, bottom, left, right):
    """
    Converts only a rectangular region of a color image to grayscale.
    """
    h, w, c = color_image.shape

    # Validate parameters
    if not (0 <= top < bottom <= h):
        print(f"Error: top ({top}) and bottom ({bottom}) must satisfy 0 <= top < bottom <= {h}")
        return None
    if not (0 <= left < right <= w):
        print(f"Error: left ({left}) and right ({right}) must satisfy 0 <= left < right <= {w}")
        return None

    # Copy image to avoid modifying the original
    output_image = color_image.copy()

    # Extract and convert region to grayscale
    region = output_image[top:bottom, left:right]
    gray_region = (0.299*region[:, :, 0] + 0.587*region[:, :, 1] + 0.114*region[:, :, 2]).astype(np.uint8)

    # Turn the 2D grayscale array into 3 channels
    output_image[top:bottom, left:right] = np.stack([gray_region]*3, axis=-1)

    return output_image


im = cv2.imread('lena.tiff')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
res = partial_grayscale(im, top=100, bottom=300, left=50, right=250)
plt.imshow(res)
plt.title("Partially grayscaled region")
plt.show()


# d) Move a black rectangle across the image (animation effect)
def blackout_rectangle(image, top, left, height, width):
    """
    Draws a black rectangle at a given position on the image.
    """
    h, w, c = image.shape
    new_image = image.copy()

    # Compute rectangle limits (stay inside bounds)
    bottom = min(top + height, h)
    right = min(left + width, w)

    # All values to 0 so it is black
    new_image[top:bottom, left:right] = 0
    return new_image


def move_rectangle(image, rect_height, rect_width, step_size=1):
    """
    Moves a black rectangle across the image step by step 
    to create an animation using matplotlib.
    """
    H, W, C = image.shape
    plt.figure()

    for top in range(0, H, step_size):
        for left in range(0, W, step_size):
            img_with_rect = blackout_rectangle(image, top, left, rect_height, rect_width)
            # Clear the last image
            plt.clf()
            plt.imshow(img_with_rect)
            plt.draw()
            plt.pause(0.01)  # short pause for animation

    plt.show()


img = cv2.imread('lena.tiff')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
move_rectangle(img, rect_height=100, rect_width=100, step_size=20)

# e) Apply a threshold mask to an image

def threshold_mask(image, threshold):
    """
    Applies a threshold to an image.
    Returns:
      - masked_image: pixels below threshold set to black
      - mask: binary mask where True = pixel >= threshold
    """
    img_float = image.astype(np.float32)

    # Convert to grayscale if multi-channel
    if img_float.ndim == 3:  # if it has 3 channels
        gray = img_float.mean(axis=2)  # average R, G, B to get grayscale
    else:
        gray = img_float.copy()

    # Create binary mask
    mask = gray >= threshold

    # Prepare output image
    masked_image = np.zeros_like(image)

    # Apply mask (preserve pixels above threshold)
    masked_image[mask] = image[mask]

    return masked_image, mask


img = cv2.imread('lena.tiff')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

thresholds = [50, 128, 200]
plt.figure(figsize=(12, 4))

for i, t in enumerate(thresholds):
    masked_img, mask = threshold_mask(img, t)

    plt.subplot(2, len(thresholds), i + 1)
    plt.imshow(masked_img)
    plt.title(f'Threshold={t}')

    plt.subplot(2, len(thresholds), i + 1 + len(thresholds))
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask')

plt.tight_layout()
plt.show()
