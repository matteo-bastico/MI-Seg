import os
import warnings
import numpy as np
import SimpleITK as sitk

from argparse import ArgumentParser
from rt_utils import RTStructBuilder

_LABEL_CODE = {
    'Femoral head_L (consensus)': 1,
    'Femoral head_R (consensus)': 2,
    'Penile bulb (consensus)': 3,
    'Prostate (consensus)': 4,
    'Rectum (consensus)': 5,
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="data/", type=str)
    args = parser.parse_args()
    path = args.data_path
    for patient_folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, patient_folder)):
            if os.path.isfile(os.path.join(path, patient_folder, file)):
                try:
                    rtstruct = RTStructBuilder.create_from(
                        dicom_series_path=os.path.join(path, patient_folder, 'T2'),
                        rt_struct_path=os.path.join(path, patient_folder, file)
                    )
                    # Loading the 3D Mask from within the RT Struct
                    mask_3d = np.zeros(rtstruct.get_roi_mask_by_name('External').shape)
                    for key in _LABEL_CODE:
                        if key in rtstruct.get_roi_names():
                            # Here we avoid also overlapping labels, it can happen with prostate and rectum
                            # a small overlap in some samples
                            mask_3d[mask_3d == 0] += _LABEL_CODE[key] * rtstruct.get_roi_mask_by_name(key)[mask_3d == 0]
                        else:
                            warnings.warn("Skipping label {} not found in {}.".format(key, patient_folder))
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(path, patient_folder, 'T2'))
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    label = sitk.GetImageFromArray((mask_3d.swapaxes(0, 2)).swapaxes(1, 2))
                    label.SetSpacing(image.GetSpacing())
                    label.SetOrigin(image.GetOrigin())
                    print(patient_folder)
                    print("Image size:", image.GetSize())
                    print("Label size:", label.GetSize())
                    sitk.WriteImage(label, os.path.join(path, patient_folder, 'labels.nii.gz'))
                except:
                    # Probably file is not an RT_struct
                    pass