import cv2
import pytesseract
import numpy as np
import PIL.Image
from PIL import Image
from sklearn.cluster import KMeans


# Returns input image
def get_original(image):
	return image

# Turns image to grayscale
def get_grayscale(image):
	return cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)

# Removes noise by blurring slightly
def remove_noise(image):
	return cv2.medianBlur(image, 5)

# Old method of separating pixels by threshold
def thresholding(image):
	ret, binary_image = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary_image


# New method of thresholding
def new_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Finding the threshold to set for the binary cutoff for the pixels
    gray_lists = gray.tolist()
    values = list(itertools.chain.from_iterable(gray_lists))
    x = np.array(values)
    model = KMeans(2).fit(x.reshape(-1,1))
    centers = model.cluster_centers_
    cutoff = centers.mean()
    
    # Converting the image to binary colors
    binary_image = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)[1]
        
    return binary_image


# Extracts text from image using pytesseract (returns list of words)
def extract_words(image):
	myconfig = "--psm 3 --oem 3"
	print(type(image))
	text = pytesseract.image_to_string(image, config = myconfig)
	lines = text.split('\n')
	words = list(filter(lambda line: line.strip() != '', lines))
	return words



functions = [get_original, get_grayscale, thresholding, cv2.bitwise_not]


image_code = 'ALYA'

# Read image
# image = cv2.imread(code + '.png')
image = PIL.Image.open(image_code + '.png')
agreed_text = list()

# Apply functions to image one by one
for function in functions:

    # Alter image and extract text
    image = function(image)
    words = extract_words(image)

    # Record new words
    for word in words:
    	if word not in agreed_text:
    		agreed_text.append(word)

print(agreed_text)



# image = cv2.bitwise_not(image)

# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image
