# Seb's Camo Recolourer

![Preview Image](https://github.com/Seb105/CamoRecolourer/blob/main/images/example1.JPG?raw=true)
## Antivirus note:

This program was written in Python and compiled using Nuitka, and therefore the standalone executable bundles the entire Python runtime.

Antivirus programs often falsely detect compiled python executables as a virus, due to this compilation process bundling the runtime.

### Running without the executable:

If you don't trust the executable due to this antivirus warning, you can download [Python](https://www.python.org/downloads/) and install the following modules using PIP:
```pip
pip install tkinter
pip install Pillow
pip install scikit-learn
```
Then, run main.py and the program will run identically to the executable.

# How to use:

## Loading an image

When a camo is loaded , it will be automatically analysed to find the dominant colours of the image.
If this automatic analysis has too many or too few colours, you can tweak the detection parameters:

### Max colours:
The maximum number of colours the algorithm will detect.

### Input colours
You can manually set the input colours if the detection misses important colours

## Getting an output

When an image is loaded, the input image will show in its unmodified form on the left, and the resulting output image on the right.

Press on a colour in output list, and select a new colour to see the the result in realtime.

Save using the button at the bottom.

## Using the colour transposer

The colour transposer attempts to match the colours from a different image (transposed image) to the existing colours in the currently loaded image (input image)

It has two modes to match colours by:
#### Brightness:
The brightest colour of the transposed image will be matched to the brightest of the input image, the darkest to the darkest etc.
#### Frequency:
The most frequent colour in the input image will be matched to the most frequent colour in the transposed image, the least frequent to the least frequent etc.
