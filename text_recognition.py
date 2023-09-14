import cv2
import pytesseract
import numpy as np
import PIL.Image
from sklearn.cluster import KMeans





# Returns input image
def get_original(image):
	return image

# Turns image to grayscale
def get_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Removes noise by blurring slightly
def remove_noise(image):
	return cv2.medianBlur(image, 5)

# Old method of separating pixels by threshold
def thresholding(image):
	ret, binary_image = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary_image


# New method of thresholding
def new_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Finding the threshold to set for the binary cutoff for the pixels
    gray_lists = gray.tolist()
    values = list(itertools.chain.from_iterable(gray_lists))
    x = np.array(values)
    model = KMeans(2).fit(x.reshape(-1,1))
    centers = model.cluster_centers_
    cutoff = centers.mean()
    
    # Converting the image to binary colors
    ret, binary_image = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
        
    return binary_image

# Extracts text from image using pytesseract (returns list of words)
def extract_words(image):
	myconfig = "--psm 3 --oem 3"
	text = pytesseract.image_to_string(PIL.Image.open(image), config = myconfig)
	lines = text.split('\n')
	words = list(filter(lambda line: line.strip() != '', lines))
	return words



functions = [get_original, get_grayscale, thresholding, cv2.bitwise_not]




# code = 'ADSW'
code = 'ALYA'

image = cv2.imread(code + '.png')
agreed_text = []

for function in functions:

	image = function(image)
	words = extract_words(image)

	for word in words:
		if word not in agreed_text:
			agreed_text.append(word)

print(agreed_text)



# image = cv2.bitwise_not(image)

# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image
