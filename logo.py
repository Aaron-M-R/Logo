import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
import cv2
import imutils
from utils import show_image
import pytesseract

pd.options.mode.chained_assignment = None


class Logo:
    # Loading in the image
    def __init__(self, path): 
        self.orig_path = path
        self.warnings = list()
        self.name = os.path.basename(path)
        
        # Reading in the image
        raw_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # Seeing if there are any transparent pixels
        if raw_data.shape[2] > 3:
            alphas = raw_data[:,:,3]
            alphas = np.ravel(alphas)
            transparent_pixels = alphas[alphas < 255]
            if len(transparent_pixels) > 0:
                #print('Warning: Transparent pixels found.  Any pure transparent pixels will be converted to standard RGB pixels with a white value.')
                self.warnings.append('Transparent Pixels Found')
                raw_data[np.where((raw_data == [0,0,0,0]).all(axis=2))] = [255,255,255,255]
        
        # Converting the image to RGB and saving the image
        image = cv2.cvtColor(raw_data, cv2.COLOR_BGRA2RGB)
        self.image = image
        
        
    # Detecting the colors with a default value of 2 colors to be found
    def color_detect(self):
        # Setting up the list of each RGB color value
        r = list()
        g = list()
        b = list()
        for line in self.image:
            for pixel in line:
                temp_r, temp_g, temp_b = pixel
                r.append(temp_r)
                g.append(temp_g)
                b.append(temp_b)
        
        # Creating a dataframe from the lists
        color_df = pd.DataFrame({'red':r, 'blue':b, 'green':g})
        test_df = (color_df==0)
        
        # Standardizing the columns using whitening
        if test_df['red'].all() != True:
            color_df['scaled_red'] = whiten(color_df['red'].astype(float))
        else:
            color_df['scaled_red'] = color_df['red'].astype(float)
            
        if test_df['blue'].all() != True:
            color_df['scaled_blue'] = whiten(color_df['blue'].astype(float))
        else:
            color_df['scaled_blue'] = color_df['blue'].astype(float)
            
        if test_df['green'].all() != True:
            color_df['scaled_green'] = whiten(color_df['green'].astype(float))
        else:
            color_df['scaled_green'] = color_df['green'].astype(float)
            
        # Finding the amount of clusters
        color_clusters = 2
        done = False
        temp_df = color_df[['scaled_red','scaled_green','scaled_blue']]
        while (done == False) and (color_clusters < 10):
            kmeans_model = KMeans(n_clusters=color_clusters, n_init="auto").fit(temp_df)
            y_kmeans = kmeans_model.predict(temp_df)
            logo_df = temp_df.copy()
            logo_df['cluster'] = y_kmeans
            distribution_df = logo_df['cluster'].value_counts().reset_index()
            distribution_df.columns = ['cluster','pixels']
            distribution_df['percentage'] = distribution_df['pixels']/len(logo_df)
            smallest_group = distribution_df['percentage'].min()
            if smallest_group < 0.05:
                color_clusters -= 1
                done = True
            else:
                color_clusters += 1
                
        if color_clusters == 1:
            color_clusters = 2
                
        clustering = KMeans(n_clusters=color_clusters, n_init="auto").fit(temp_df)
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
        self.colors = color_weights
        
    
    def image_to_binary(self, input_image, print_steps=False):
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        if print_steps==True:
            show_image(gray)
        
        # Finding the threshold to set for the binary cutoff for the pixels
        gray_lists = gray.tolist()
        values = list(itertools.chain.from_iterable(gray_lists))
        x = np.array(values)
        model = KMeans(2).fit(x.reshape(-1,1))
        centers = model.cluster_centers_
        cutoff = centers.mean()
        
        # Converting the image to binary colors
        ret, binary_image = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
        if print_steps==True:
            show_image(binary_image)
            
        self.binary = binary_image
            
################################################################################

    def extract_text(self, print_steps=False):
        # Loading in the image (why must it be resized?)
        resized = imutils.resize(self.image, width=300)
        
        # Blurring the image
        blurred = cv2.GaussianBlur(resized, (5,5), 0)
        if print_steps==True:
            show_image(blurred)

        # Setting all pixels to white or black
        self.image_to_binary(blurred, print_steps)
        if print_steps==True:
            show_image(self.binary)

        # Search for text in image and return as list of words
        myconfig = "--psm 3 --oem 3"
        text = pytesseract.image_to_string(self.binary, config = myconfig)
        lines = text.split('\n')
        text = list(filter(lambda line: line.strip() != '', lines))

        return text

################################################################################


    def shape_analysis(self, print_steps=False):
        # Loading in the image
        resized = imutils.resize(self.image, width=300)
        
        # Processing the image
        blurred = cv2.GaussianBlur(resized, (5,5), 0)
        if print_steps==True:
            show_image(blurred)

        self.image_to_binary(blurred, print_steps)
        
        # Contouring the image
        contours, hierarchy = cv2.findContours(self.binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
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
       
        if print_steps==True:
            show_image(image_with_contours)
        
        # Setting and returning the values from our analysis
        self.contour_count = contour_count
        self.contour_area = contour_area
        self.contour_points = contour_points
        self.image_with_contours = image_with_contours
        
        
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
    

    # If the class is printed, this will return the read in path
    def __str__(self):
        return self.name
        
    
    # If the class is simply called, it will print the image of the logo
    def __repr__(self):
       return self.name

    