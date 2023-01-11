import os
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
        # TODO: change + with os.path.join
        for file in os.listdir(path + patient_folder):
            if os.path.isfile(path + patient_folder + '/' + file):
                try:
                    rtstruct = RTStructBuilder.create_from(
                        dicom_series_path=path + patient_folder + "/T2",
                        rt_struct_path=path + patient_folder + '/' + file
                    )
                    # Loading the 3D Mask from within the RT Struct
                    mask_3d = np.zeros(rtstruct.get_roi_mask_by_name('External').shape)
                    for key in _LABEL_CODE:
                        mask_3d += _LABEL_CODE[key] * rtstruct.get_roi_mask_by_name(key)
                    reader = sitk.ImageSeriesReader()
                    dicom_names = reader.GetGDCMSeriesFileNames(path + patient_folder + "/T2")
                    reader.SetFileNames(dicom_names)
                    image = reader.Execute()
                    label = sitk.GetImageFromArray((mask_3d.swapaxes(0, 2)).swapaxes(1, 2))
                    label.SetSpacing(image.GetSpacing())
                    label.SetOrigin(image.GetOrigin())
                    print(patient_folder)
                    print("Image size:", image.GetSize())
                    print("Label size:", label.GetSize())
                    sitk.WriteImage(label, path + patient_folder + '/labels.nii.gz')
                except:
                    pass
