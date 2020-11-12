# sprint-diffimage-ml
scrum sprint for ML transient detection/classification in difference images


Product Goal: a general-purpose difference image analysis pipeline that can identify multiply-imaged transients using machine learning. 

Inputs: difference images as FITS files.  Associated source catalogs from the static-sky (template) images and the difference images. 

Outputs: categorization score for any transient candidates in the image using three classes
0 - not a real transient
1 - single transient
2 - multiply-imaged transient candidate

