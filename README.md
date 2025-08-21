# Stop-Sign-Detection
Detect and isolate bright red regions in an image using OpenCV (HSV color masking).
Stop Sign Detection with OpenCV

This project detects STOP signs in images using Python + OpenCV.
The pipeline is based on color masking in the HSV color space, contour detection, and polygon approximation.
Detected stop signs are highlighted with a green rectangle, and their center coordinates are calculated.

#Features
Detects bright red regions (HSV thresholding).
Approximates contours and finds octagonal shapes (8 points).
Draws bounding boxes and labels around detected STOP signs.
Resizes processed images for display.
Saves output images (result1.jpg, result2.jpg, …) automatically to stop_sign_dataset_output/.

#Requirements
Python version: 3.10.7
OpenCV version: 4.12.0
NumPy version: 2.1.2 

#Usage
Place your input images inside "stop_sign_dataset"
Create folder named "stop_sign_dataset_output" rocessed images will appear in windows and be saved in this folder.
