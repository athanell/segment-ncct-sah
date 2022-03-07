# Inference
This section decribes how to use the algorithm to segment SAH using your images.

## Data preparation
1. If your images are in a dicom format, they should be structured in the following way:
    ```console
    dicoms_dir/
    ├── subjectID1
    │   ├── img001.dcm
    │   └── img002.dcm
    │   ...
    │   └── img00N.dcm
    │
    ├── subjectID2
    │   ├── img001.dcm
    │   └── img002.dcm
    |    ...
    │   └── img00N.dcm
    │
    │...
    │
    └── subjectIDN
        ├── img001.dcm
        └── img002.dcm
        ...
        └── img00N.dcm
    ```
    If your images are in any other format: a) convert them to nifti, b) preprocess them with your own software to meet the objectives of step 2 and skip executing step 2. 

2. Run the preprocessing step that will convert the dicom images to nifti format, resample them to 256&times;256 (in-plane resolution) and clip the intensity units to 0&ndash;150 HU (so that values above 150 HU will be set to 150 HU and values below 0 HU will be set to 0 HU).
    ```console
    user@ubuntu:~$ python preprocess.py <dicoms_dir> <niftis_dir>
    ```
## SAH segmentation
1. Add the image path with the niftis to `job/config.ini` as the `path_to_search` value:

        [img]
        path_to_search = <niftis_dir>

2. Run the inference script that will segment SAH
    ```console
    user@ubuntu:~$ python run.py --task inference --config job/config.ini --job-dir job
    ```

