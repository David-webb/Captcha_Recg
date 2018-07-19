#!/usr/bin/python
# -*- coding: utf-8 -*-

# Created by David Teng on 18-7-6
import pytesseract
from PIL import Image

image = Image.open("testrm.jpg")
code = pytesseract.image_to_string(image)
print code

