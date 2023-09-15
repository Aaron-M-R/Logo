from logo import Logo
import PIL.Image
from utils import show_image
import logo_comparison
from logo_comparison import text_similarity


image_code = 'BBBY'


image = Logo('/Users/aaronrasin/Desktop/Logo/logos/' + image_code + '.png')




words = logo_comparison.text_similarity(image)

print("words: ")
print(words)

