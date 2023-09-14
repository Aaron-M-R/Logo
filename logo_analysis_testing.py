import pandas as pd
import numpy as np
import os
import logo
from logo import Logo
from logo_comparison import color_data
import logo_comparison
from tqdm import tqdm


# Loading the data
applicant_loc = r'C:/Users/aaronrasin/Desktop/Logo'
applicant_logo_names = os.listdir(applicant_loc)

previous_loc = r'C:/Users/aaronrasin/Desktop/Logo'
previous_logo_names = os.listdir(previous_loc)

applicant_logos = list()
previous_logos = list()

for i in applicant_logo_names:
    applicant_logos.append(Logo(applicant_loc + '\\' + i))
    
for i in previous_logo_names:
    previous_logos.append(Logo(previous_loc + '\\' + i))

# Getting the scores
ssim_df = pd.DataFrame()
color_df = pd.DataFrame()
sc_df = pd.DataFrame()
tm_df = pd.DataFrame()

applicant_list = list()
previous_list = list()
ssim_score_list = list()
color_score_list = list()
tm_score_list = list()

for applicant in applicant_logos:
    applicant.color_detect()
    for previous in tqdm(previous_logos):
        ssim = logo_comparison.logo_ssim(applicant, previous)
        
        previous.color_detect()
        color = logo_comparison.calculate_color_similarity(applicant, previous)
        tm_score = logo_comparison.logo_contains(applicant, previous)
        
        applicant_list.append(applicant.name)
        previous_list.append(previous.name)
        ssim_score_list.append(ssim)
        color_score_list.append(color)
        tm_score_list.append(tm_score)
    sc_df = pd.concat([sc_df, logo_comparison.calculate_logo_shape_complexity_similarity(applicant,previous_logos)])
    
data_df = pd.DataFrame({'Applicant Logo':applicant_list,
                        'Previous Logo':previous_list,
                        'SSIM':ssim_score_list,
                        'Color Similarity Score':color_score_list,
                        'Template Matching':tm_score_list})
sc_df.columns = ['Applicant Logo','Previous Logo','Shape Complexity Score']
data_df = data_df.merge(sc_df, how='inner', on=['Applicant Logo','Previous Logo'])
data_df = data_df[['Applicant Logo','Previous Logo','SSIM','Color Similarity Score','Shape Complexity Score','Template Matching']]
data_df.to_excel(r'C:\Users\nscop\Documents\Education\PhD\Logo Analysis\Code\Logo_Analysis\example_images\full_example\full_example_scores.xlsx', index=False)




