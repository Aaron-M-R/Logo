import pandas as pd
from word_image import create_word_image

save_loc = '/Users/Desktop/Logo/Company Name Word Pics'
font_path = '/Users/Desktop/Logo/fonts/ARIAL.TTF'

data_df = pd.read_csv('/Users/Desktop/Logo/companies.csv')

company_names = data_df['company name'].unique().tolist()

for company in company_names:
    if len(company) < 50:
        create_word_image(company, font_path, save_loc + company.replace('/','').replace('*','').replace('?','').replace('.','') + '.png')
        
        