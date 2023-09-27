import pandas as pd
import numpy as np
import os
from logo import Logo
from logo_comparison import color_data
import logo_comparison
from tqdm import tqdm
from fuzzywuzzy import fuzz


# Loading the data
applicant_loc = '/Users/aaronrasin/Desktop/Logo/Testing/previous'
applicant_logo_names = os.listdir(applicant_loc)

applicant_logos = list()
text_list = dict()

for name in applicant_logo_names:
	text_list[name] = logo_comparison.text_similarity(Logo(applicant_loc + '/' + name))

for logo in text_list.items():
	print(logo)




	print(max(fuzz.token_set_ratio(logo[0], logo[1][0]), fuzz.token_set_ratio(logo[0], logo[1][1])))

