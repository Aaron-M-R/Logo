import numpy as np
import pandas as pd

import logo
import imutils
import cv2

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from utils import crop_center
from fuzzywuzzy import fuzz


# FUNCTIONS

    # image_resize

    # logo_ssim
    # calculate_color_similarity
    # text_similarity
    # calculate_logo_shape_complexity_similarity
    # logo_contains

    # compare_logos



# Resizing images to equal dimensions
def image_resize(imageA, imageB):
    # Making sure the images are equal size, necessary for some comparison methods
    if (imageA.shape[0] != imageB.shape[0]) or (imageA.shape[1] != imageB.shape[1]):
        imageB = cv2.resize(imageB,(imageA.shape[1],imageA.shape[0]))
    return imageA, imageB


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


def calculate_color_similarity(logoA, logoB, print_full_results=False):
    color_analysis = dict()

    primary_difference = deltaE_ciede2000(logoA.primary_iw,logoB.primary_iw)[0][0]
    secondary_difference = deltaE_ciede2000(logoA.secondary_iw,logoB.secondary_iw)[0][0]
    color_analysis['Standard With White'] = [primary_difference, secondary_difference]

    # primary_difference = deltaE_ciede2000(logoA.primary_ew,logoB.primary_ew)[0][0]
    # secondary_difference = deltaE_ciede2000(logoA.secondary_ew,logoB.secondary_ew)[0][0]
    # color_analysis['Standard Without White'] = [primary_difference, secondary_difference]
    
    # primary_difference = deltaE_ciede2000(logoA.primary_ew,logoB.primary_iw)[0][0]
    # secondary_difference = deltaE_ciede2000(logoA.secondary_ew,logoB.secondary_iw)[0][0]
    # color_analysis['Standard Without White A, With White B'] = [primary_difference, secondary_difference]
    
    # primary_difference = deltaE_ciede2000(logoA.primary_iw,logoB.primary_ew)[0][0]
    # secondary_difference = deltaE_ciede2000(logoA.secondary_iw,logoB.secondary_ew)[0][0]
    # color_analysis['Standard Without White B, With White A'] = [primary_difference, secondary_difference]

    primary_difference = deltaE_ciede2000(logoA.secondary_iw,logoB.primary_iw)[0][0]
    secondary_difference = deltaE_ciede2000(logoA.primary_iw,logoB.secondary_iw)[0][0]
    color_analysis['Standard With White Flipped'] = [primary_difference, secondary_difference]
    
    # primary_difference = deltaE_ciede2000(logoA.secondary_ew,logoB.primary_ew)[0][0]
    # secondary_difference = deltaE_ciede2000(logoA.primary_ew,logoB.secondary_ew)[0][0]
    # color_analysis['Standard Without White Flipped'] = [primary_difference, secondary_difference]
    
    # primary_difference = deltaE_ciede2000(logoA.secondary_ew,logoB.primary_iw)[0][0]
    # secondary_difference = deltaE_ciede2000(logoA.primary_ew,logoB.secondary_iw)[0][0]
    # color_analysis['Standard Without White A, With White B Flipped'] = [primary_difference, secondary_difference]
    
    # primary_difference = deltaE_ciede2000(logoA.secondary_iw,logoB.primary_ew)[0][0]
    # secondary_difference = deltaE_ciede2000(logoA.primary_iw,logoB.secondary_ew)[0][0]
    # color_analysis['Standard Without White B, With White A Flipped'] = [primary_difference, secondary_difference]

    # Combining the data
    color_analysis_df = pd.DataFrame(color_analysis).T.reset_index()
    color_analysis_df.columns = ['Method','Score 1','Score 2']    
    color_analysis_df['Full Score'] = color_analysis_df[['Score 1','Score 2']].max(axis=1)
    
    if print_full_results == True:
        print(color_analysis_df)
    
    return color_analysis_df['Full Score'].min()



# Return similarity score given two logos using the already extracted text
def text_similarity(logoA, logoB):

    text_scores = list()

    for textA in logoA.text:
        for textB in logoB.text:
            score = fuzz.token_set_ratio(textA, textB)
            text_scores.append(score)

    for contoursA in logoA.contours:
        for contoursB in logoB.contours:
            score = cv2.matchShapes(contoursA, contoursB, method=1, parameter=0)
            score = score/7
            text_scores.append(score)

    return max(text_scores)



# Contour analysis comparisons
def calculate_logo_shape_complexity_similarity(logoA,logoList):
    # Making sure not to repeat any logos
    if logoA not in logoList:
        logoList.append(logoA)
    
    # Extracting the data
    data = dict()
    for logo in logoList:
        # if hasattr(logo, 'contour_count') == False:
        #     logo.shape_analysis()
        
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
    similarity_df = similarity_df.drop(similarity_df.shape[0]-1)
    
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

    

# Logo Comparison
def compare_logos(applicant_logos, previous_logos):
    
    # Creating dataframes and lists to record the scores
    ssim_df = pd.DataFrame()
    color_df = pd.DataFrame()
    sc_df = pd.DataFrame()
    tm_df = pd.DataFrame()
    text_df = pd.DataFrame()

    applicant_list = list()
    previous_list = list()
    ssim_score_list = list()
    color_score_list = list()
    tm_score_list = list()
    text_score_list = list()
    truth_value_list = list()
    
    # Comparing logos
    print("Comparing logos")
    for applicant in tqdm(applicant_logos):
        for previous in previous_logos:

            # Calculate Similarity Scores (assign floats)
            ssim = logo_ssim(applicant, previous)
            color = calculate_color_similarity(applicant, previous)
            tm_score = logo_contains(applicant, previous)
            text_score = text_similarity(applicant, previous)

            # Record Similarity Scores (append floats to list)
            applicant_list.append(applicant.name)
            previous_list.append(previous.name)
            ssim_score_list.append(ssim)
            color_score_list.append(color)
            tm_score_list.append(tm_score)
            text_score_list.append(text_score)

        sc_df = pd.concat([sc_df, calculate_logo_shape_complexity_similarity(applicant,previous_logos)])
        
        # Counteract SCS function adding an applicant to the previous list
        previous_logos = previous_logos[:-1]
        
    # Construct DataFrame
    data_df = pd.DataFrame({'Applicant Logo':applicant_list,
                        'Previous Logo':previous_list,
                        'SSIM':ssim_score_list,
                        'Color Similarity Score':color_score_list,
                        'Template Matching':tm_score_list,
                        'Text Similarity Score':text_score_list})
    sc_df.columns = ['Applicant Logo','Previous Logo','Shape Complexity Score']
    data_df = data_df.merge(sc_df, how='inner', on=['Applicant Logo','Previous Logo'])
    data_df = data_df[['Applicant Logo','Previous Logo','SSIM','Color Similarity Score','Shape Complexity Score','Template Matching', 'Text Similarity Score']]

    return data_df


# Record to excel
# data_df.to_excel('/Users/aaronrasin/Desktop/Logo/LogoComparisonData.xlsx', index=False)

