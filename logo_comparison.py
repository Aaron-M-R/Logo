import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
import imutils
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from utils import crop_center
import logo
from fuzzywuzzy import fuzz

# Resizing images to equal dimensions
def image_resize(imageA, imageB):
    # Making sure the images are equal size, necessary for some comparison methods
    if (imageA.shape[0] != imageB.shape[0]) or (imageA.shape[1] != imageB.shape[1]):
        imageB = cv2.resize(imageB,(imageA.shape[1],imageA.shape[0]))
    return imageA, imageB


# Mean Square Error
def logo_mse(logoA, logoB):
    # Reading the logo attributes
    imageA = logoA.image
    imageB = logoB.image
    
    # Resizing if necessary
    imageA, imageB = image_resize(imageA, imageB)
    
    # Calculating MSE
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Structural similarity Image measure
def logo_ssim(logoA, logoB):
    # Reading the logo attributes
    imageA = logoA.image
    imageB = logoB.image
    
    # Resizing if necessary
    imageA, imageB = image_resize(imageA, imageB)
    
    # Calculating MSE
    ssim_score = ssim(imageA, imageB, win_size = 3, multichannel=True)
    return ssim_score


# Calculating the color similarity based off the primary and secondary colors
class color_data():
    def __init__(self, RGB_colors):
        # Extracting the RGB dataframe
        rgb_df = pd.DataFrame()
        for key, value in RGB_colors.items():
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
        self.rgb_df = rgb_df
        
        # Seeing if the top color is white
        if int(rgb_df.values.tolist()[0][4]) == 1:
            self.top_color_white = True
            self.top_white_coverage = rgb_df.values.tolist()[0][3]
        else:
            self.top_color_white = False
            self.top_white_coverage = 0
            
    def return_top_labs(self, top_df):
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

    def return_top_colors(self, include_white=True):
        # Seeing whether or not to filter and then doing so
        if (len(self.rgb_df) == 2) or (include_white == True):
            top_df = self.rgb_df
        else:
            top_df = self.rgb_df[self.rgb_df['White']!=1]
        
        # Splitting to the top two colors
        top_df['Weighted Coverage'] = top_df['Coverage']/top_df['Coverage'].sum()
        
        return self.return_top_labs(top_df)


def calculate_color_similarity(logoA, logoB, print_full_results=False):
    color_analysis = dict()
   
    # Loading the data
    logoA_cd = color_data(logoA.colors)
    logoB_cd = color_data(logoB.colors)
    
    # Analyzing the primary colors regardless of white
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=True)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=True)
    
    primary_difference = deltaE_ciede2000(logoA_primary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_secondary,logoB_secondary)[0][0]
    
    color_analysis['Standard With White'] = [primary_difference, secondary_difference]
    
    # Analyzing the primary colors without white
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=False)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=False)
    
    primary_difference = deltaE_ciede2000(logoA_primary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_secondary,logoB_secondary)[0][0]
    
    color_analysis['Standard Without White'] = [primary_difference, secondary_difference]
    
    # Analyzing A without white
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=False)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=True)
    
    primary_difference = deltaE_ciede2000(logoA_primary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_secondary,logoB_secondary)[0][0]
    
    color_analysis['Standard Without White A, With White B'] = [primary_difference, secondary_difference]
    
    # Analyzing B without white
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=True)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=False)
    
    primary_difference = deltaE_ciede2000(logoA_primary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_secondary,logoB_secondary)[0][0]
    
    color_analysis['Standard Without White B, With White A'] = [primary_difference, secondary_difference]

    # Analyzing the primary colors regardless of white, flipped colors
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=True)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=True)
    
    primary_difference = deltaE_ciede2000(logoA_secondary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_primary,logoB_secondary)[0][0]
    
    color_analysis['Standard With White Flipped'] = [primary_difference, secondary_difference]
    
    # Analyzing the primary colors without white, flipped colors
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=False)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=False)
    
    primary_difference = deltaE_ciede2000(logoA_secondary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_primary,logoB_secondary)[0][0]
    
    color_analysis['Standard Without White Flipped'] = [primary_difference, secondary_difference]
    
    # Analyzing A without white, flipped colors
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=False)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=True)
    
    primary_difference = deltaE_ciede2000(logoA_secondary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_primary,logoB_secondary)[0][0]
    
    color_analysis['Standard Without White A, With White B Flipped'] = [primary_difference, secondary_difference]
    
    # Analyzing B without white, flipped colors
    logoA_primary, logoA_secondary, logoA_primary_weight, logoA_secondary_weight =  logoA_cd.return_top_colors(include_white=True)
    logoB_primary, logoB_secondary, logoB_primary_weight, logoB_secondary_weight =  logoB_cd.return_top_colors(include_white=False)
    
    primary_difference = deltaE_ciede2000(logoA_secondary,logoB_primary)[0][0]
    secondary_difference = deltaE_ciede2000(logoA_primary,logoB_secondary)[0][0]
    
    color_analysis['Standard Without White B, With White A Flipped'] = [primary_difference, secondary_difference]

    # Combining the data
    color_analysis_df = pd.DataFrame(color_analysis).T.reset_index()
    color_analysis_df.columns = ['Method','Score 1','Score 2']    
    color_analysis_df['Full Score'] = color_analysis_df[['Score 1','Score 2']].max(axis=1)
    
    if print_full_results == True:
        print(color_analysis_df)
    
    return color_analysis_df['Full Score'].min()
    


################################################################################

# Text analysis
def text_similarity(logoA, logoB):

    text_scores = list()

    for textA in logoA.text:
        for textB in logoB.text:
            text_scores.append(fuzz.token_set_ratio(textA, textB))

    return max(text_scores)

def find_truth(applicant, previous):

    # Remove .png/.jpeg from names
    if '.' in previous.name:
        previous_name = previous.name.split('.')[0]
    if '.' in applicant.name:
        applicant_name = applicant.name.split('.')[0]

    # Name is no longer than 5 characters
    if len(previous_name) > 5:
        previous_name = previous_name[:4]

    # Return 1 if names match
    if previous_name in applicant_name:
        return 1
    return 0

################################################################################  



# Contour analysis comparisons
def calculate_logo_shape_complexity_similarity(logoA,logoList):
    # Making sure not to repeat any logos
    if logoA not in logoList:
        logoList.append(logoA)
    
    # Extracting the data
    data = dict()
    for logo in logoList:
        if hasattr(logo, 'contour_count') == False:
            logo.shape_analysis()
        
        contour_count = logo.contour_count
        contour_area = logo.contour_area
        contour_points = logo.contour_points
        name = logo.name
        values = [contour_count, contour_area, contour_points]
        data[name] = values
    
    # Creating a dataframe to be used for analysis    
    data_df = pd.DataFrame.from_dict(data, orient='index')
    data_df['Image Name'] = data_df.index
    data_df.columns = ['Contour Count','Contour Area','Contour Points','Image Name']
    data_df = data_df.reset_index(drop=True)
    
    # Scaling the data
    scaler = MinMaxScaler()
    data_df['Scaled Contour Count'] = scaler.fit_transform(data_df['Contour Count'].values.reshape(-1,1))
    data_df['Scaled Contour Area'] = scaler.fit_transform(data_df['Contour Area'].values.reshape(-1,1))
    data_df['Scaled Contour Points'] = scaler.fit_transform(data_df['Contour Points'].values.reshape(-1,1))
    
    # Calculating similarity scores
    similarity_data = list()
    for index_1, row_1 in data_df.iterrows():
        image_1 = logoA.name
        image_2 = row_1['Image Name']
        values_1 = data_df[data_df['Image Name'] == image_1][['Scaled Contour Count','Scaled Contour Area','Scaled Contour Points']].to_numpy().flatten()
        values_2 = data_df[data_df['Image Name'] == image_2][['Scaled Contour Count','Scaled Contour Area','Scaled Contour Points']].to_numpy().flatten()

        similarity = euclidean(values_1,values_2)
        temp = [image_1, image_2, similarity]
        similarity_data.append(temp)
            
    # Cleaning the similarity scores dataframe
    similarity_df = pd.DataFrame(similarity_data, columns=['Logo 1','Logo 2','Similarity Score'])
    similarity_df = similarity_df[similarity_df['Logo 1']!=similarity_df['Logo 2']]
    similarity_df['Similarity Score'] = scaler.fit_transform(similarity_df['Similarity Score'].values.reshape(-1,1))
    
    similarity_df['Similarity Score'] = (similarity_df['Similarity Score']-1).abs()
    similarity_df = similarity_df.sort_values(by=['Similarity Score'], ascending=False)
    
    return similarity_df


# Seeing if one image is in another image
def logo_contains(applicant_logo, previous_logo, threshold=0.7, use_center=False, show_results=False):
    # Open the template and convert it to edges while extracing its shape
    applicant_raw = applicant_logo.image.copy()
    applicant_gray = cv2.cvtColor(applicant_raw, cv2.COLOR_RGB2GRAY)

    if use_center==True:
        applicant_gray = crop_center(applicant_gray)
    applicant_edges = cv2.Canny(applicant_gray, 100, 150)
    (applicant_height, applicant_width) = applicant_raw.shape[:2]
    
    # Open the previous logo
    previous_raw = previous_logo.image.copy()
    previous_grey = cv2.cvtColor(previous_raw, cv2.COLOR_RGB2GRAY)
    temp_found = None

    for scale in np.linspace(0.1, 0.1+(0.05*78), 79)[::-1]:

       # Resize the previous logo to various sizes relative to the scale
       previous_resized = imutils.resize(previous_grey, width=int(previous_grey.shape[1]*scale))
       ratio = previous_grey.shape[1] / float(previous_resized.shape[1])

       # If the size of the applicant is larger than the size of the previous, stop
       if previous_resized.shape[0] < applicant_height or previous_resized.shape[1] < applicant_width:
          break

       # Converting to edges
       previous_edges = cv2.Canny(previous_resized, 100, 150)

       # Running the match and extracting relevant statistics
       match = cv2.matchTemplate(previous_edges, applicant_edges, cv2.TM_CCOEFF_NORMED)
       (_, val_max, _, loc_max) = cv2.minMaxLoc(match)

       # Saving the top found amount
       if temp_found is None or val_max>temp_found[0]:
          temp_found = (val_max, loc_max, ratio)

    # Setting the match score
    if temp_found is None:
        match_score = 0
    else:
        match_score = temp_found[0]
        
    # Seeing whether or not to show the image
    if show_results == True and match_score >= threshold:
        (_, loc_max, r) = temp_found
        (x_start, y_start) = (int(loc_max[0]), int(loc_max[1]))
        (x_end, y_end) = (int((loc_max[0] + applicant_height)), int((loc_max[1] + applicant_width)))
        # Showing where on the image the applicant was found
        cv2.rectangle(previous_raw, (x_start, y_start), (x_end, y_end), (153, 51, 255), 5)
        cv2.imshow('Template Found', previous_raw)
        cv2.waitKey(0)
    else:
        pass
    
    return match_score