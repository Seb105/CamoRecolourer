# Seb's Camo Recolourer

![Preview Image](https://github.com/Seb105/CamoRecolourer/blob/main/images/example1.JPG?raw=true)
## Antivirus note:

This program was written in Python and compiled using Nuitka, and therefore the standalone executable bundles the entire Python runtime.

Antivirus programs often falsely detect compiled python executables as a virus as a consequence of how it is compiled

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

## Getting an output

When an image is loaded, two identical images of the camo will load.
The detected colours will show on the right as both input and output

Press on a colour in output list, and select a new colour to see the the result in realtime.

Save using the button at the bottom.
