from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Setting the values
# shape_files = os.listdir(file_loc)

data_dict = dict()

# Loading in the files
# for i in shape_files:
#     data_dict[i] = cv2.imread(file_loc + i,0)


# Setting the function to load the contours
def get_contours(input_image, show_image=False):
    # Start Aaron's code!
    shape = cv2.cvtColor(input_image.astype('uint8'), cv2.COLOR_BGR2GRAY)
    # End Aaron's code
    _, binary = cv2.threshold(shape, 225, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    image = cv2.drawContours(input_image.astype('uint8'), contours, -1, (0, 255, 0), 2)
    
    if show_image == True:
        plt.imshow(image)
        plt.show()
        
        plt.imshow(binary, cmap="gray")
        plt.show()
    
    return contours[0]

def best_method(img):
    # Finding the best method
    image_a = list()
    image_b = list()
    similarity = list()
    methods = list()


    # Double loop through every word image
    for filea, imagea in data_dict.items():
        for fileb, imageb in data_dict.items():
            for method in [1,2,3]:
                image_a.append(filea)
                image_b.append(fileb)
                score = cv2.matchShapes(get_contours(imagea),
                                        get_contours(imageb),
                                        method,0)
                similarity.append(score)
                methods.append(method)
            
    wsh_df = pd.DataFrame({'Image A':image_a,
                            'Image B':image_b,
                            'Score':similarity,
                            'Method':methods})


    # Analyzing the methods
    wsh_df = wsh_df.sort_values(by=['Method','Score'], ascending=[True, True])

    for i in wsh_df['Method'].unique():
        temp_df = wsh_df.loc[wsh_df['Method']==i]
        pivot_df = temp_df.pivot('Image A',columns='Image B')[['Score']]
        sns.heatmap(pivot_df, annot=True).set_title(f'Method {i} Heatmap')
        plt.show()

    return wsh_df    
    
