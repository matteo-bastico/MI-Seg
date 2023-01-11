import os
import warnings
import SimpleITK as sitk

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="data/", type=str)
    parser.add_argument("--mask_image", type=float, help='Mask image path for the bias correction')
    parser.add_argument("--shrink_factor", default=2, type=float, help='Shrink factor for the bias correction')
    parser.add_argument("--number_fitting_levels", default=4, type=int, help='Mask image path for the bias correction')
    parser.add_argument("--maximum_iterations", type=int, help='Mask image path for the bias correction')
    args = parser.parse_args()
    path = args.data_path
    for patient_folder in os.listdir(path):
        # TODO: change + with os.path.join
        inputImage = sitk.ReadImage(path + patient_folder + "/T2.nii.gz", sitk.sitkFloat32)
        image = inputImage
        if args.mask_image:
            maskImage = sitk.ReadImage(args.mask_image, sitk.sitkUInt8)
        else:
            maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        if args.shrink_factor > 1:
            image = sitk.Shrink(
                inputImage, [int(args.shrink_factor)] * inputImage.GetDimension()
            )
            maskImage = sitk.Shrink(
                maskImage, [int(args.shrink_factor)] * inputImage.GetDimension()
            )
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if args.maximum_iterations:
            corrector.SetMaximumNumberOfIterations(
                [args.maximum_iterations] * args.number_fitting_levels
            )
        corrected_image = corrector.Execute(image, maskImage)
        log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
        corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
        sitk.WriteImage(corrected_image_full_resolution, path + patient_folder + '/T2_N4.nii.gz')
        print(patient_folder, " Conversion completed.")
