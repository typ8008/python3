# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:10:17 2020

Prerequisit:
    Require Pillow to export to jpg

example of bar code generator

@author: Mariusz
"""

from barcode import EAN13, PZN
from barcode.writer import ImageWriter
from io import BytesIO
from wand.image import Image

# print to a file-like object:
rv = BytesIO()
PZN(str(100000902922), writer=ImageWriter()).write(rv)
EAN13(str(100000902922), writer=ImageWriter()).write(rv)


# or sure, to an actual file:
with open('barcodePZN.jpeg', 'wb') as f:
    PZN('100000011111', writer=ImageWriter()).write(f)

with open('barcodeEAN13.jpeg', 'wb') as f:
    EAN13('100000011111', writer=ImageWriter()).write(f)

  
# Save as SVG using wand (image Magick)
import barcode
#from wand.image import Image

EAN = barcode.get_barcode_class('ean13')
#ean = EAN(u'123456789011', writer=ImageWriter())
ean = EAN(u'123456789011', writer=Image())

ean = barcode.get('ean13','123456789102', writer=Image() )
filename = ean.save('barcode_test')
#fullname = ean.save('my_ean13_barcode')    