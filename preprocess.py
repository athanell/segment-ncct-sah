import os
import argparse
import warnings
import numpy as np
import nilearn.image
import nibabel as nib
import dicom2nifti

warnings.filterwarnings("ignore")

def main():
    input_dir, output_dir = get_args()
    print('Processing...')
    os.makedirs(output_dir, exist_ok=True)
    for patient_id in sorted(next(os.walk(input_dir))[1]):
        print('   ' + patient_id)
        img_dir = os.path.join(input_dir, patient_id)
        img = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, output_file=None, reorient_nifti=False)['NII']
        preprocess_nifti(img).to_filename(os.path.join(output_dir, patient_id + '.nii.gz'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory containing dicom image directories')
    parser.add_argument('output', help='output directory for nifti files')
    args = parser.parse_args()
    return os.path.expanduser(args.input), os.path.expanduser(args.output)

def preprocess_nifti(img, downsample=2, size=256, clip=(0, 150), dtype=np.int16):
    new_affine = img.affine.copy()
    new_affine[:3, :3] = np.matmul(img.affine[:3, :3], np.diag((downsample, downsample, 1)))
    min_value = img.get_fdata().min()
    tmp_img = nilearn.image.resample_img(img, target_affine=new_affine,
        target_shape=(size, size, img.shape[2]), fill_value=min_value)
    data = tmp_img.get_fdata()
    if clip:
        data = data.clip(min=clip[0], max=clip[1])
    return nib.Nifti1Image(data.astype(dtype), tmp_img.affine)

if __name__ == '__main__':
    main()
