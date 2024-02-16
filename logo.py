import os
import itertools
import cv2
import imutils
import pytesseract
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
from utils import show_image, crop_center
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from PIL import Image, ImageDraw, ImageFont, ImageChops

pd.options.mode.chained_assignment = None


class Logo:
    # Loading in the image
    def __init__(self, path): 


        # INITIAL SETUP

        # Setting up path, name and warning list
        self.orig_path = path
        self.warnings = list()
        self.name = os.path.basename(path)
        
        # Reading in logo image
        self.raw_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Check for presence of transparent pixels in logo image
        self.check_transparency()



        # COLOR ANALYSIS

        # Converting logo image to RGB and saving
        self.image = cv2.cvtColor(self.raw_data, cv2.COLOR_BGRA2RGB)

        # Store colors found in logo image
        self.colors = self.color_detect()

        # Extract colors of logo image as RGB (DataFrame)
        self.rgb_df = self.extract_RGB()

        # Detect if white is top color of logo image
        self.top_white_coverage = self.check_top_white()

        # Store two most dominant colors of logo image (iw=include white, ew=exclude_white)
        self.primary_iw, self.secondary_iw, self.primary_weight_iw, self.secondary_weight_iw = self.return_top_colors(include_white=True)
        # self.primary_ew, self.secondary_ew, self.primary_weight_ew, self.secondary_weight_ew = self.return_top_colors(include_white=False)



        # SHAPE COMPLEXITY ANALYSIS

        # Store logo contour info
        self.contour_count, self.contour_area, self.contour_points, self.image_with_contours = self.shape_analysis()



        # TEXT ANALYSIS

        # Read text off logo image
        self.text = self.extract_text()

        # Store contour info found in text
        self.contours = self.convert_to_contours()

        






    # GENERAL FUNCTIONS

    # Seeing if there are any transparent pixels
    def check_transparency(self):
        if self.raw_data.shape[2] > 3:
            alphas = self.raw_data[:,:,3]
            alphas = np.ravel(alphas)
            transparent_pixels = alphas[alphas < 255]
            if len(transparent_pixels) > 0:
                #print('Warning: Transparent pixels found.  Any pure transparent pixels will be converted to standard RGB pixels with a white value.')
                self.warnings.append('Transparent Pixels Found')
                self.raw_data[np.where((self.raw_data == [0,0,0,0]).all(axis=2))] = [255,255,255,255]


    # Convert image to binary (black or white pixels)
    def image_to_binary(self, print_steps=False):

        # Ignore KMeans warnings
        warnings.filterwarnings("ignore")

        # Convert to grayscale
        gray = cv2.cvtColor(self.blurred, cv2.COLOR_RGB2GRAY)
        # if print_steps==True:
        #     show_image(gray)
        
        # Finding the threshold to set for the binary cutoff for the pixels
        gray_lists = gray.tolist()
        values = list(itertools.chain.from_iterable(gray_lists))
        x = np.array(values)
        model = KMeans(n_clusters=2, n_init='auto').fit(x.reshape(-1,1))
        centers = model.cluster_centers_
        cutoff = centers.mean()
        
        # Converting the image to binary colors
        ret, binary_image = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
        # if print_steps==True:
        #     show_image(binary_image)
            
        return binary_image








    # SHAPE COMPLEXITY FUNCTIONS

    # Returns contour info about logo (count, area, points, image)
    def shape_analysis(self, print_steps=False):
        # Loading in the image
        resized = imutils.resize(self.image, width=300)
        
        # Processing the image
        self.blurred = cv2.GaussianBlur(resized, (5,5), 0)
        # if print_steps==True:
        #     show_image(self.blurred)

        binary_image = self.image_to_binary(self.blurred)
        
        # Contouring the image
        contours, hierarchy = cv2.findContours(binary_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyzing the countours
        contour_count = len(contours)
        contour_area = 0
        contour_points = 0
        image_with_contours = resized.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            contour_area += area
            contour_points += len(cnt)
            if area > 5:
                approx = cv2.approxPolyDP(cnt, 0.009*cv2.arcLength(cnt, True), True) 
           
                # Checking if the no. of sides of the selected region is 7. 
                cv2.drawContours(image_with_contours, [approx], 0, (0, 0, 255), 5) 
       
        # if print_steps==True:
        #     show_image(image_with_contours)

        return contour_count, contour_area, contour_points, image_with_contours

    






    # TEXT ANALYSIS

    # Read text on logo image and store it in logo object
    def extract_text(self, print_steps=False):
        # Loading in the image (why must it be resized?)
        resized = imutils.resize(self.image, width=300)
        
        # Blurring the image
        self.blurred = cv2.GaussianBlur(resized, (5,5), 0)
        # if print_steps==True:
        #     show_image(self.blurred)

        # Setting all pixels to white or black
        binary_image = self.image_to_binary()

        # Search for text in image and return as list of words
        myconfig = "--psm 3 --oem 3"
        text = pytesseract.image_to_string(binary_image, config = myconfig)

        return text


    # Turn given words into images written in given font
    def create_word_image(self, font_path = '/Users/Desktop/Logo/fonts/ARIAL.TTF'):
        # Creating the image outline
        img = Image.new('RGB', (1000,75), color=(255,255,255))
        d = ImageDraw.Draw(img)
        
        # Setting up the font
        font = ImageFont.truetype(font_path, 36)
        d.text((20,20), self.text, font=font, fill=(0,0,0))
        
        # Cropping the image
        imageBox = img.getbbox()
        img = img.crop(imageBox)
        
        return img


    # Return letter types of letters in given words
    def find_letter_types(self):
        # Figure out which types of letters are in the word
        lower_letters = ['g','j','p','q','y']
        upper_letters = ['b','d','f','h','i','j','k','l','t']
        capital_letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
                           'P','Q','R','S','T','U','V','W','X','Y','Z']
        
        lower_letters_contained = len(set(self.text).intersection(set(lower_letters)))
        upper_letters_contained = len(set(self.text).intersection(set(upper_letters)))
        capital_letters_contained = len(set(self.text).intersection(set(capital_letters)))
        
        contains_lower = (lower_letters_contained > 0)
        contains_upper = (upper_letters_contained > 0)
        contains_capital = (capital_letters_contained > 0)
        contains_regular = (lower_letters_contained + contains_upper + capital_letters_contained != len(self.text))

        return contains_lower, contains_upper, contains_capital, contains_regular


    # Clean a given image
    def clean_image(self):
        img = self.word_image

        # Crop image
        img_copy = Image.new(img.mode, img.size, (255,255,255))
        diff = ImageChops.difference(img, img_copy)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            new_img = img.crop(bbox)
        else:
            new_img = img
        
        # Converting the image to binary
        thresh = 223
        bin_img = cv2.threshold(np.array(new_img.copy()), thresh, 255, cv2.THRESH_BINARY)[1]
        
        # Used to also return new img (still available)
        return bin_img


    # Returns image as shapes
    def convert_to_shape(self):  

        # These may have to be adjusted depending on the font chosen
        upper_lim = 7
        lower_lim = 19
        gray_shade = 128
        
        # We use different levels within the image depending on the input letters
        contains_lower, contains_upper, contains_capital, contains_regular = self.letter_types
        input_image = self.bin_img

        # All regular letters
        if (contains_lower==False) and (contains_upper==False) and (contains_capital==False):
            input_image = np.full(input_image.shape,gray_shade)
        
        # Has upper or capital letters
        elif (contains_lower==False) and (contains_upper==True or contains_capital==True):
            for row in range(0,upper_lim):
                row_data = input_image[row]
                for column in range(len(row_data)):
                    if row_data[column].all() == 0:
                        input_image[:,column] = gray_shade
            input_image[upper_lim:] = gray_shade

        # Contains lower letters
        elif (contains_lower==True) and (contains_upper==False) and (contains_capital==False):
            for row in range(lower_lim,input_image.shape[0]):
                row_data = input_image[row]
                for column in range(len(row_data)):
                    if row_data[column].all() == 0:
                        input_image[:,column] = gray_shade
            input_image[:lower_lim] = gray_shade

        else:
            for row in range(0,upper_lim):
                row_data = input_image[row]
                for column in range(len(row_data)):
                    if row_data[column].all() == 0:
                        input_image[:upper_lim,column] = gray_shade
            for row in range(lower_lim+upper_lim,input_image.shape[0]):
                row_data = input_image[row]
                for column in range(len(row_data)):
                    if row_data[column].all() == 0:
                        input_image[lower_lim+upper_lim:input_image.shape[0],column] = gray_shade
            input_image[upper_lim:lower_lim+upper_lim] = gray_shade
            
        return input_image


    # Setting the function to load the contours
    def get_contours(self, show_image=False):
        input_image = self.shapes

        # Convert image to gray shade, then binary, then contours
        shape = cv2.cvtColor(input_image.astype('uint8'), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(shape, 225, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        image = cv2.drawContours(input_image.astype('uint8'), contours, -1, (0, 255, 0), 2)
        
        if show_image == True:
            plt.imshow(image)
            plt.show()
            
            plt.imshow(binary, cmap="gray")
            plt.show()
        
        return contours[0]


    # Create and store shapes from text stored in logo object
    def convert_to_contours(self):

        self.word_image = self.create_word_image()
        self.letter_types = self.find_letter_types()
        self.bin_img = self.clean_image()
        self.shapes = self.convert_to_shape()
        contours = self.get_contours()
        return contours








    # COLOR ANALYSIS FUNCTIONS

    # Detecting the colors with a default value of 2 colors to be found
    def color_detect(self):

        # Ignore warnings (mostly from KMeans)
        warnings.filterwarnings("ignore")

        # Setting up the list of each RGB color value
        r = list()
        g = list()
        b = list()
        for line in self.image:
            for pixel in line:
                if sum(pixel) < 225*3:
                    temp_r, temp_g, temp_b = pixel
                    r.append(temp_r)
                    g.append(temp_g)
                    b.append(temp_b)
        
        # Creating a dataframe from the lists
        color_df = pd.DataFrame({'red':r, 'blue':b, 'green':g})
        # test_df = (color_df==0)

        # Standardizing the columns using whitening
        try:
            # if test_df['red'].all() != True:
            color_df['scaled_red'] = whiten(color_df['red'].astype(float))
        except:
            color_df['scaled_red'] = color_df['red'].astype(float)
            
        try:
            # if test_df['blue'].all() != True:
            color_df['scaled_blue'] = whiten(color_df['blue'].astype(float))
        except:
            color_df['scaled_blue'] = color_df['blue'].astype(float)
            
        try:
            # if test_df['green'].all() != True:
            color_df['scaled_green'] = whiten(color_df['green'].astype(float))
        except:
            color_df['scaled_green'] = color_df['green'].astype(float)

        # Finding the amount of clusters
        color_clusters = 2
        done = False
        temp_df = color_df[['scaled_red','scaled_green','scaled_blue']]

        while (done == False) and (color_clusters < 5):
            kmeans_model = KMeans(n_clusters=color_clusters, n_init='auto').fit(temp_df)
            y_kmeans = kmeans_model.predict(temp_df)
            logo_df = temp_df.copy()
            logo_df['cluster'] = y_kmeans
            distribution_df = logo_df['cluster'].value_counts().reset_index()
            distribution_df.columns = ['cluster','pixels']
            distribution_df['percentage'] = distribution_df['pixels']/len(logo_df)
            smallest_group = distribution_df['percentage'].min()
            if smallest_group < 0.1:
                color_clusters -= 1
                done = True
            else:
                color_clusters += 1
                
        if color_clusters == 1:
            color_clusters = 2

        clustering = KMeans(n_clusters=color_clusters, n_init = 'auto').fit(temp_df)
        cluster_centers = clustering.cluster_centers_
        color_df['cluster'] = clustering.predict(temp_df)
        
        colors = list()
        r_std, g_std, b_std = color_df[['red','green','blue']].std()
        
        # Extracting the centers from the clusters
        colors = list()
        for cluster_center in cluster_centers:
            scaled_r, scaled_g, scaled_b = cluster_center
            colors.append(((scaled_r*r_std)/255,
                           (scaled_g*g_std)/255,
                           (scaled_b*b_std)/255))
        
        # Assigning the value to the class
        rgb_colors = (np.array(colors) * 255).astype(int)
        rgb_colors = rgb_colors.tolist()
        rgb_colors = [tuple(x) for x in rgb_colors]    
        
        # Getting the color values and weights
        color_weights = dict()
        for i in range(len(rgb_colors)):
            color_weights[rgb_colors[i]] = len(color_df[color_df['cluster']==i])/len(color_df)

        return color_weights
    

    # Extract the RGB info as a DataFrame
    def extract_RGB(self):
        rgb_df = pd.DataFrame()
        for key, value in self.colors.items():
            R = key[0]
            G = key[1]
            B = key[2]
            coverage = value
            row = [R,G,B,coverage]
            temp_df = pd.DataFrame(row).transpose()
            rgb_df = pd.concat([rgb_df,temp_df])
        rgb_df.columns = ['Red','Green','Blue','Coverage']
        rgb_df = rgb_df.sort_values(by=['Coverage'], ascending=False)
        rgb_df['White'] = np.where((rgb_df['Red']>240) & (rgb_df['Blue']>240) & (rgb_df['Green']>240),1,0)
        return rgb_df


    # Seeing if the top color of logo image is white (returns top white coverage)
    def check_top_white(self):
        if int(self.rgb_df.values.tolist()[0][4]) == 1:
            self.top_color_white = True
            return self.rgb_df.values.tolist()[0][3]
        else:
            self.top_color_white = False
            return 0
        

    # Return two most dominant colors in LAB format
    def return_top_colors(self, include_white=True):
        # Seeing whether or not to filter and then doing so
        if (len(self.rgb_df) == 2) or (include_white == True):
            top_df = self.rgb_df
        else:
            top_df = self.rgb_df[self.rgb_df['White']!=1]
        
        # Splitting to the top two colors
        top_df['Weighted Coverage'] = top_df['Coverage']/top_df['Coverage'].sum()

        # Extracing the data from that dataframe and converting the colors to LAB colorspace
        primary_color_data = top_df.values.tolist()[0]
        primary_color_rgb = [primary_color_data[0]/255,primary_color_data[1]/255,primary_color_data[2]/255]
        primary_color_weighted_coverage = primary_color_data[5]
        primary_color_lab = rgb2lab([[primary_color_rgb]])
        
        if len(top_df.values.tolist()) == 1:
            secondary_color_data = top_df.values.tolist()[0]
        else:
            secondary_color_data = top_df.values.tolist()[1]
        secondary_color_rgb = [secondary_color_data[0]/255,secondary_color_data[1]/255,secondary_color_data[2]/255]
        secondary_color_weighted_coverage = secondary_color_data[5]
        secondary_color_lab = rgb2lab([[secondary_color_rgb]])
        
        return primary_color_lab, secondary_color_lab, primary_color_weighted_coverage, secondary_color_weighted_coverage








    # DEVELOPMENT FUNCTIONS

    # Building the funcionality to easily call the image
    def show_image(self):
        plt.imshow(self.image, 'gray')
        plt.xticks([]), plt.yticks([])
        plt.show()
        

    # Show the colors in a matplotlib plot, if colors are not yet detected will also run that  
    def show_colors(self):
        if hasattr(self, 'colors') == False:
            self.color_detect()
        plt.imshow([list(self.colors.keys())])
        plt.xticks([]), plt.yticks([])
        plt.show()
    

    # If the class is printed, tshow_colorshis will return the read in path
    def __str__(self):
        return self.name
        
    
    # If the class is simply called, it will print the image of the logo
    def __repr__(self):
       return self.name


    