from fpdf import FPDF
import os
from sys import platform

DARWIN = (platform == "darwin")
slash = '/' if DARWIN else '\\'

def create_il_from_epson(start, end, BASE_DIR):
    imagelist = []
    if end < start:
        print("End must be bigger than or equal to start! ")
    else:
        for i in range(start, end+1):
            filename = ""
            if 0<=i<10:
                filename = BASE_DIR+"\EPSON00"+str(i)+".JPG"
            elif i<100:
                filename = BASE_DIR+"\EPSON0" + str(i) + ".JPG"
            imagelist.append(filename)
    return imagelist


def change_slash(string):
    result = ""
    for i in range(0, len(string)):
        if string[i] == "\\":
            result += "/"
        else:
            result += string[i]
    return result


# BASE_DIR = change_slash("C:\\Users\\Jack\\Desktop\\Courses\\CSJT189 Discrete Differential Geometry\\HW4")
BASE_DIR = "C:/Users/Jack/Desktop/Courses/CSJT189 Discrete Differential Geometry/HW4"
os.chdir(BASE_DIR)
# imagelist is the list with all image filenames
imagelist = create_il_from_epson(7, 9, BASE_DIR)

pdf = FPDF()
# pdf = FPDF(orientation = 'L') #if you want your pdf to be in Landscape mode
pdf.set_auto_page_break(0)

for image in imagelist:
    pdf.add_page()
    pdf.image(image,w=190)
pdf.output("HW.pdf", "F")