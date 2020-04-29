# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:12:20 2020

Prerequisits:
    Install Ghost script https://www.ghostscript.com/download/gsdnld.html,
    Install ImageMagick https://imagemagick.org/script/download.php

example showing how to convert pdf to png file.

@author: Mariusz
"""


from __future__ import print_function

from wand.image import Image

with Image(filename='Payslip.pdf', resolution = 800) as img:

    print('width =', img.width)
    print('height =', img.height)
    print('pages = ', len(img.sequence))
    print('resolution = ', img.resolution)

    with img.convert('png') as converted:
        converted.save(filename='sample_doc.png')
