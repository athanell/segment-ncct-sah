# Subarachnoid haemorrhage segmentation from non-contrast head ct images

<img src="/instructions/media/Demo.png" width="591px" height='591' />

## Description 
This repository is based on the work described in the article  

 ```
 Development and External Validation of a DeepLearning Algorithm to Identify and Localize Subarachnoid Hemorrhage on CT Scans 
``` 
and locates (semantic segmentation) subarachnoid blood (SAH) from non-contrast head ct scans (NCCT). 

It is a U-net convolutional neural network that was trained using 90 multi-plannar reformat (MPR) image volumes of patients with SAH and 22 image volumes of controls without any intracranial blood findings. 

This algorithm is expected to perform robustly throughout a wide quality range of images (pre-operative, post-operative). Typically post-operative images exhibit intense artifacts.

## Details
The software is a modified version of the [NiftyNet](https://github.com/NifTK/NiftyNet). It can be used for inference, where the user provides as input their NCCT volumes and subarachnoid blood will be segmented. The result is a nifti binary 3D volume mask where 1 represents blood and 0 background. 

 

## Performance 
The patient-level and slice-level performance of the algorithm was measured on two external validation sets of images from [India](http://headctstudy.qure.ai/#dataset) (279 ncct volumes) and Switzerland (1100 ncct volumes) as well as on consecutive emergency scans from five Finnish hospitals (519 ncct volumes). Voxel-level performance was measured on a set of scans from Finnish hospitals (49 ncct volumes).  


* External validation sets (non Finnish)

  * Patient-level

    |                             | Swiss (SAH)| Swiss (controls)|India (SAH)|India (controls)|
    |-----------------------------|------------|-----------------|-----------|----------------|
    |Cases                        |100  	     |1000  	         |37	       |242             |      
    |Predicted SAH                |100	       |423	             |36         |34              |
    
    |                             | Swiss |India|
    |-----------------------------|-------|-----|
    |Sensitivity (%)              |100.0  |97.3 |
    |Specificity (%)              |57.7	  |86.0 |
    |Precision (%)                |19.1	  |51.4 |
    |Negative predictive value (%)|100.0	|99.5 |
    |False positive rate (%)      |42.3	  |14.0 |
    |False discovery rate (%)     |80.9	  |48.6 |
    |False negative rate (%)      |0.0	  |2.7  |
    |Accuracy (%)                 |61.5	  |87.5 |


  * Slice-level

    |                             | Swiss (SAH)| Swiss (controls)|
    |-----------------------------|------------|-----------------|
    |Slices                       |2110	  	  |46954  	         |     
    |Predicted SAH                |1845		    |2200	             |
    
    |                             | Swiss 
    |-----------------------------|-------|
    |Sensitivity (%)              |87.4   |
    |Specificity (%)              |95.3	  |
    |Precision (%)                |45.6	  |
    |Negative predictive value (%)|99.4  	|
    |False positive rate (%)      |4.7	  |
    |False discovery rate (%)     |54.4	  |
    |False negative rate (%)      |12.6	  |
    |Accuracy (%)                 |95.0	  |
  
    The [Indian](http://headctstudy.qure.ai/#dataset) dataset did not have slice-level annotations
  
* External validation set (Finland)
 
  * Patient-level

    |                             | Finnish (SAH)| Finnish (controls)|
    |-----------------------------|----|-----------------------------|
    |Cases                        |8   |511   	                     |    
    |Predicted SAH                |8   |65	                         |
    
    |                             |Finland|
    |-----------------------------|-------|
    |Sensitivity (%)              |100.0  |
    |Specificity (%)              |87.3	  |
    |Precision (%)                |11.0	  |
    |Negative predictive value (%)|100.0	|
    |False positive rate (%)      |12.7	  |
    |False discovery rate (%)     |89.0	  |
    |False negative rate (%)      |0.0	  |
    |Accuracy (%)                 |87.5	  |

  * Slice-level 
    |                             | Finnish (SAH)| Finnish (controls)|
    |-----------------------------|----|-----------------------------|
    |Cases                        |77  |27090   	                   |    
    |Predicted SAH                |58  |329	                         |
    
    |                             |Finland|
    |-----------------------------|-------|
    |Sensitivity (%)              |75.3   |
    |Specificity (%)              |98.8	  |
    |Precision (%)                |15.0	  |
    |Negative predictive value (%)|99.9 	|
    |False positive rate (%)      |1.2	  |
    |False discovery rate (%)     |85.0	  |
    |False negative rate (%)      |24.7	  |
    |Accuracy (%)                 |98.7	  |
  


## Assumptions
* The NCCT data can be in DICOM or in a NIFTI format. See more details in [data preparation](instructions/01-inference-unet.md)


## Installation 
Start by [installing the necessary prerequisites ](instructions/00-prerequisites.md) , and then test the algorithm by using the trained network to [segment SAH using your images](instructions/01-inference-unet.md).

## Contents
The following structure is adopted

| File/folder       | Description                                |
|-------------------|--------------------------------------------|
| `instructions\`   | A step-by-step guide on how to install & use this software |
