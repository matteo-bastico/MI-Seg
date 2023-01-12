import os
import warnings
import SimpleITK as sitk

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="data/", type=str)
    args = parser.parse_args()
    path = args.data_path
    for patient_folder in os.listdir(path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(path, patient_folder, 'T2'))
        reader.SetFileNames(dicom_names)
        image_t2 = reader.Execute()
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(path, patient_folder, 'Deformed_CT'))
        reader.SetFileNames(dicom_names)
        image_ct = reader.Execute()
        if image_t2.GetSpacing() != image_ct.GetSpacing():
            warnings.warn("Difference spacing found in volumes for {}: T2 {}, CT {}. "
                          "Setting spacing of T2 to deformed CT".format(
                image_t2.GetSpacing(),
                image_ct.GetSpacing(),
                patient_folder
            ))
            # Set CT spacing as T2, to avoid problems in training for label spacing
            image_ct.SetSpacing(image_t2.GetSpacing())
        sitk.WriteImage(image_ct, os.path.join(path, patient_folder, '/Deformed_CT.nii.gz'))
        sitk.WriteImage(image_t2, os.path.join(path, patient_folder, '/T2.nii.gz'))
