import matplotlib.pyplot as plt
import cv2


def show_image(image):
    plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
def image_percentage_resize(image, scale_percent):
    #calculate the 50 percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    
    # dsize
    dsize = (width, height)
    
    # resize image
    output = cv2.resize(image, dsize)
    return output


def crop_center(image):
    w3 = int(image.shape[0]/3)
    h3 = int(image.shape[1]/3)
    cropped_image = image[h3:h3*2, w3:w3*2]
    return cropped_image