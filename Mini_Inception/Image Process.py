from PIL import Image
import numpy as np
from itertools import chain

def flatten(listOfLists):
    "Flatten one level of nesting"
    return list(chain.from_iterable(listOfLists))

X = range(0,601,40)
i = 0

lists = [[(num,y) for num in X] for y in range(0,401,40)]

lol = flatten(lists)
i = 0
for el in lol:
    img = Image.open('Picture.jpg', 'r')
    background = Image.new('RGBA', (80, 80), (150, 150, 150, 255))
    bg_w, bg_h = background.size
    img.paste(background, el)
    img.save(str(i)+'out.png')
    i+=1