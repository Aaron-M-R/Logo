import pandas as pd
from word_image import *

save_loc = '/Users/Desktop/Logo/Company_Name_Word_Pics'
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'

def generate(words)
# words = ['Hello', 'Mello', 'Trickster', 'Pleasant', 'Unbelievable']

    for word in words:
        img = create_word_image(word, font_path)
        new_img, bin_img = clean_image(img)
        shape_image = convert_to_shape(bin_img.copy(), gray_shade, contains_lower, contains_upper, contains_capital, contains_regular)

        cv2.imwrite(save_loc + word + '_original.png', new_img) 
        cv2.imwrite(save_loc + word + '_shape.png', shape_image) 


# data_df = pd.read_csv('/Users/Desktop/Logo/companies.csv')

# company_names = data_df['company name'].unique().tolist()

# for company in company_names:
#     if len(company) < 50:
#         create_word_image(company, font_path, save_loc + company.replace('/','').replace('*','').replace('?','').replace('.','') + '.png')
        
#         