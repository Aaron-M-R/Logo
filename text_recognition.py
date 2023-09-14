import cv2
import pytesseract
import numpy as np
import PIL.Image


def ocr_core(image):
	text = pytesseract.image_to_string(image)
	return text

def get_original(image):
	return image

def get_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
	return cv2.medianBlur(image, 5)

def thresholding(image):
	return cv2.threshold(image, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def list_of_words(text):
    lines = text.split('\n')
    words = list(filter(lambda line: line.strip() != '', lines))
    return words

def pyread(image):
	myconfig = "--psm 3 --oem 3"
	text = pytesseract.image_to_string(PIL.Image.open(image), config = myconfig)
	return text



#print(text)





# def printer(mode, image):
# 	lines = ocr_core(image)
# 	print(mode + ": ", end="")
# 	for line in lines:
# 		if len(line) > 0:
# 			print(line, end = "")

# lines = []

functions = [get_original, get_grayscale, thresholding]




# code = 'ADSW'
code = 'ALYA'
image = cv2.imread('/Users/aaronrasin/Desktop/Logo/logos/' + code + '.png')
agreed_text = []

for function in functions:
	image = function(image)
	text = pyread(image)
	words = list_of_words(text)
	for word in words:
		if word not in agreed_text:
			agreed_text.append(word)
print("Round 1:")
print(agreed_text)


agreed_text = []

for function in functions:
	image = function(image)
	text = ocr_core(image)
	words = list_of_words(text)
	for word in words:
		if word not in agreed_text:
			agreed_text.append(word)

print("Round 2:")
print(agreed_text)


# print(lines)
# cv2.imshow('test', image)

# # image = remove_noise(image)
# # print("Regular: " + ocr_core(image))

# #cv2.imshow('test', image)

# image = cv2.bitwise_not(image)

# print("Crazy Idea: " + pytesseract.image_to_string(image, config = myconfig))

# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image
