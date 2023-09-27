from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 

def create_word_image(words, font_path, save_locs):
    # Creating the image outline
    img = Image.new('RGB', (1000,75), color=(255,255,255))
    d = ImageDraw.Draw(img)
    
    # Setting up the font
    font = ImageFont.truetype(font_path, 36)
    d.text((20,20), words, font=font, fill=(0,0,0))
    
    # Cropping the image
    imageBox = img.getbbox()
    img = img.crop(imageBox)
    
    # Creating a temp file or saving the file depending on the user input
    img.save(save_loc)
    
    return


def clean_image(save_loc):
    img = cv2.imread(save_loc,0)
    
    # Filtering out the tops and bottom of the image
    short_img = img[np.amin(img, axis=1) < 5]
    
    # Filtering out the leading and trailing spaces
    column_filter = np.amin(short_img, axis=0) < 5
    last_column = column_filter.shape[0] - 1 - (column_filter[::-1]!=0).argmax()
    new_img = short_img[:,np.argmax(column_filter):last_column]
    
    # Converting the image to binary
    thresh = 223
    bin_img = cv2.threshold(new_img.copy(), thresh, 255, cv2.THRESH_BINARY)[1]
    
    return new_img, bin_img


def find_letter_types(words):
    # Figure out which types of letters are in the word
    lower_letters = ['g','j','p','q','y']
    upper_letters = ['b','d','f','h','i','j','k','l','t']
    capital_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
                       'P','Q','R','S','T','U','V','W','X','Y','Z']
    
    lower_letters_contained = len(set(words).intersection(set(lower_letters)))
    upper_letters_contained = len(set(words).intersection(set(upper_letters)))
    capital_letters_contained = len(set(words).intersection(set(capital_letters)))
    
    contains_lower = (lower_letters_contained > 0)
    contains_upper = (upper_letters_contained > 0)
    contains_capital = (capital_letters_contained > 0)
    contains_regular = (lower_letters_contained + contains_upper + capital_letters_contained != len(words))

    return contains_lower, contains_upper, contains_capital, contains_regular


def convert_to_shape(input_image, gray_shade, contains_lower, contains_upper, contains_capital, contains_regular):  
    # These may have to be adjusted depending on the font chosen
    upper_lim = 7
    lower_lim = 19
    
    # We use different levels within the image depending on the input letters
    if (contains_lower==False) and (contains_upper==False) and (contains_capital==False):
        input_image = np.full(input_image.shape,gray_shade)
    elif (contains_lower==False) and (contains_upper==True or contains_capital==True):
        for row in range(0,upper_lim):
            row_data = input_image[row]
            for column in range(len(row_data)):
                if row_data[column] == 0:
                    input_image[:,column] = gray_shade
        input_image[upper_lim:] = gray_shade
    elif (contains_lower==True) and (contains_upper==False) and (contains_capital==False):
        for row in range(lower_lim,input_image.shape[0]):
            row_data = input_image[row]
            for column in range(len(row_data)):
                if row_data[column] == 0:
                    input_image[:,column] = gray_shade
        input_image[:lower_lim] = gray_shade
    else:
        for row in range(0,upper_lim):
            row_data = input_image[row]
            for column in range(len(row_data)):
                if row_data[column] == 0:
                    input_image[:upper_lim,column] = gray_shade
        for row in range(lower_lim+upper_lim,input_image.shape[0]):
            row_data = input_image[row]
            for column in range(len(row_data)):
                if row_data[column] == 0:
                    input_image[lower_lim+upper_lim:input_image.shape[0],column] = gray_shade
        input_image[upper_lim:lower_lim+upper_lim] = gray_shade
        
    return input_image




font_path = '/Users/Desktop/Logo/fonts/ARIAL.TTF'
save_loc = '/Users/Desktop/Logo/image.jpg'

gray_shade = 128


create_word_image(words, font_path, save_loc)
new_img, bin_img = clean_image(save_loc)
os.remove(save_loc)
contains_lower, contains_upper, contains_capital, contains_regular = find_letter_types(words)
shape_image = convert_to_shape(bin_img.copy(), gray_shade, contains_lower, contains_upper, contains_capital, contains_regular)


plt.imshow(new_img, cmap='gray')
plt.imshow(shape_image, cmap='gray')

save_loc = r'C:\Users\nscop\OneDrive\PhD\Research\Logo_Analysis\example_images\word_shape\\'
cv2.imwrite(save_loc + f'{words}_original.png', new_img) 
cv2.imwrite(save_loc + f'{words}_shape.png', shape_image) 
