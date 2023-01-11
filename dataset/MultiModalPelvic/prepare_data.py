import os
import shutil
import pydicom
import warnings

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="data/", type=str)
    args = parser.parse_args()
    path = args.data_path
    for patient_folder in os.listdir(path):
        dcm_T1 = []
        dcm_T2 = []
        dcm_CT = []
        dcm_deformed_CT = []
        dcm_reg = []
        dcm_rs = []
        # TODO: change + with os.path.join
        if not os.path.exists(path + patient_folder + '/T1'):
            os.mkdir(path + patient_folder + '/T1')
        if not os.path.exists(path + patient_folder + '/T2'):
            os.mkdir(path + patient_folder + '/T2')
        if not os.path.exists(path + patient_folder + '/CT'):
            os.mkdir(path + patient_folder + '/CT')
        if not os.path.exists(path + patient_folder + '/Deformed_CT'):
            os.mkdir(path + patient_folder + '/Deformed_CT')
        for file_name in sorted(os.listdir(path + patient_folder)):
            if os.path.isfile(os.path.join(path, patient_folder, file_name)) and 'labels' not in file_name:
                dcm = pydicom.read_file(os.path.join(path, patient_folder, file_name))
                if 'MR' in file_name:
                    if 'T1' in dcm['SeriesDescription'].value.upper():
                        shutil.move(os.path.join(path, patient_folder, file_name),
                                    path + patient_folder + '/T1/' + file_name)
                        dcm_T1.append(dcm)
                    elif 'T2' in dcm['SeriesDescription'].value.upper():
                        shutil.move(os.path.join(path, patient_folder, file_name),
                                    path + patient_folder + '/T2/' + file_name)
                        dcm_T2.append(dcm)
                    else:
                        warnings.warn("Found an MRI dcm that is neither T1 nor T2, it has been skipped.")
                elif 'CT' in file_name:
                    if 'Deformed' in dcm['SeriesDescription'].value:
                        shutil.move(os.path.join(path, patient_folder, file_name),
                                    path + patient_folder + '/Deformed_CT/' + file_name)
                        dcm_deformed_CT.append(dcm)
                    else:
                        shutil.move(os.path.join(path, patient_folder, file_name),
                                    path + patient_folder + '/CT/' + file_name)
                        dcm_CT.append(dcm)
                elif 'REG' in file_name:
                    # Deformable registration details as elastix parameter file
                    dcm_reg.append(dcm)
                elif 'RS' in file_name:
                    # Structures as DICOM RT STRUCT
                    dcm_rs.append(dcm)
        print(patient_folder)
        print("Found {} T1, {} T2, {} CT and {} deformed CT slices.".format(len(dcm_T1), len(dcm_T2), len(dcm_CT),
                                                                            len(dcm_deformed_CT)))
        print("Found {} REG and {} RS.".format(len(dcm_reg), len(dcm_rs)))
