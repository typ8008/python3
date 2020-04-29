# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:42:32 2020

Prerequisits:
    Uninstall PIL and Install Pillow. It's a fork of PIL. THey cannot coexit

example showing how to get text from scanned document in .PNG format
can be combined with pdfpngconverter.py

@author: Mariusz
"""


from PIL import Image
#import PIL.Image

from pytesseract import image_to_string
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'
output = pytesseract.image_to_string(Image.open('payslip.PNG').convert("RGB"), lang='eng')
print (output)