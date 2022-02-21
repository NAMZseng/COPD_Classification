import sys
import argparse
import logging
from lungmask import mask
from lungmask import lungmask_utils
import os
import SimpleITK as sitk
import numpy as np

from pathos.multiprocessing import ProcessingPool as Pool


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main(input, output, modelpath):
    # version = pkg_resources.require("lungmask")[0].version

    parser = argparse.ArgumentParser()
    # parser.add_argument('input', metavar='input', type=path, help='Path to the input image, can be a folder for dicoms')
    # parser.add_argument('output', metavar='output', type=str, help='Filepath for output lungmask')
    parser.add_argument('--modeltype', help='Default: unet', type=str, choices=['unet'], default='unet')
    parser.add_argument('--modelname', help="spcifies the trained model, Default: R231", type=str,
                        choices=['R231', 'LTRCLobes', 'LTRCLobes_R231', 'R231CovidWeb'], default='R231')
    # parser.add_argument('--modelpath', help="spcifies the path to the trained model", default=None)
    parser.add_argument('--classes', help="spcifies the number of output classes of the model", default=3)
    parser.add_argument('--cpu', help="Force using the CPU even when a GPU is available, will override batchsize to 1",
                        action='store_true')
    parser.add_argument('--nopostprocess',
                        help="Deactivates postprocessing (removal of unconnected components and hole filling",
                        action='store_true')
    parser.add_argument('--noHU',
                        help="For processing of images that are not encoded in hounsfield units (HU). E.g. png or jpg images from the web. Be aware, results may be substantially worse on these images",
                        action='store_true')
    parser.add_argument('--batchsize', type=int,
                        help="Number of slices processed simultaneously. Lower number requires less memory but may be slower.",
                        default=20)
    # parser.add_argument('--version', help="Shows the current version of lungmask", action='version', version=version)

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    batchsize = args.batchsize
    if args.cpu:
        batchsize = 1

    # logging.info(f'Load model')

    # input_image = lungmask_utils.get_input_image(args.input)
    input_image = lungmask_utils.get_input_image(input)
    # logging.info(f'Infer lungmask')
    if args.modelname == 'LTRCLobes_R231':
        # assert args.modelpath is None, "Modelpath can not be specified for LTRCLobes_R231 mode"
        assert modelpath is None, "Modelpath can not be specified for LTRCLobes_R231 mode"
        result = mask.apply_fused(input_image, force_cpu=args.cpu, batch_size=batchsize,
                                  volume_postprocessing=not (args.nopostprocess), noHU=args.noHU)
    else:
        # model = mask.get_model(args.modeltype, args.modelname, args.modelpath, args.classes)
        model = mask.get_model(args.modeltype, args.modelname, modelpath, args.classes)
        result = mask.apply(input_image, model, force_cpu=args.cpu, batch_size=batchsize,
                            volume_postprocessing=not (args.nopostprocess), noHU=args.noHU)

    if args.noHU:
        # file_ending = args.output.split('.')[-1]
        file_ending = output.split('.')[-1]
        print(file_ending)
        if file_ending in ['jpg', 'jpeg', 'png']:
            result = (result / (result.max()) * 255).astype(np.uint8)
        result = result[0]

    result_out = sitk.GetImageFromArray(result)
    result_out.CopyInformation(input_image)

    # output_dir = os.path.split(args.output)[0]
    output_dir = os.path.split(output)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # logging.info(f'Save result to: {args.output}')
    logging.info(f'Save result to: {output}')
    # sys.exit(sitk.WriteImage(result_out, args.output))
    sitk.WriteImage(result_out, output)


def mask_dcms(data_dir_name):
    """
    获取原始dcm图像的肺实质区域标签，并将标签保存为dicm格式
    :param data_dir_name:
    :return:
    """
    modelpath = '/home/MHISS/zengnanrong/COPD/checkpoint/unet_r231-d5d2fc3d.pth'
    input_root_path = "/data/zengnanrong/CTDATA/"
    output_root_path = "/data/zengnanrong/R231/"

    path = os.path.join(input_root_path, data_dir_name)
    for root, dirs, files in os.walk(path):
        for item in files:
            if '.dcm' in item.lower():
                input = os.path.join(root, item)
                output = input.replace(input_root_path, output_root_path)
                main(input, output, modelpath)


def mask_nii(patient_dir_name):
    """
    获取原始dcm图像的肺实质区域标签，并将标签保存为nii格式
    :param patient_dir_name:
    :return:
    """
    modelpath = '/home/MHISS/zengnanrong/COPD/checkpoint/unet_r231-d5d2fc3d.pth'
    input_root_path = "/data/zengnanrong/COPD_Biphasic/stage_3/DICOM"

    patient_dir_path = os.path.join(input_root_path, patient_dir_name)
    ct_dirs = get_ct_dirs(patient_dir_path)
    for dir in ct_dirs:
        input = os.path.join(patient_dir_path, dir)
        file_list = os.listdir(input)
        if len(file_list) < 2:
            continue

        output = input.replace('COPD_Biphasic', 'COPD_Biphasic_R231')
        output = output.replace('DICOM', 'nii')
        output = os.path.join(output, os.path.split(output)[1] + '.nii')
        main(input, output, modelpath)


def get_ct_dirs(root_path):
    ct_dir = []
    for item in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, item)):
            ct_dir.append(item)

    ct_dir.sort()

    return ct_dir


def segement_dcms(data_dir_name):
    ct_root_path = "/data/zengnanrong/CTDATA/"
    mask_root_path = "/data/zengnanrong/R231/"
    output_root_path = "/data/zengnanrong/LUNG_SEG/"

    path = os.path.join(ct_root_path, data_dir_name)
    for root, dirs, files in os.walk(path):
        for item in files:
            if '.dcm' in item.lower():
                ct_path = os.path.join(root, item)
                mask_path = ct_path.replace(ct_root_path, mask_root_path)

                ct_image = sitk.ReadImage(ct_path)
                ct_image_array = np.squeeze(sitk.GetArrayFromImage(ct_image))
                mask_image = sitk.ReadImage(mask_path)
                mask_image_array = np.squeeze(sitk.GetArrayFromImage(mask_image))

                height = ct_image_array.shape[0]
                width = ct_image_array.shape[1]

                for h in range(height):
                    for w in range(width):
                        if mask_image_array[h][w] == 0:
                            # 将非肺区域置0
                            ct_image_array[h][w] = 0

                seg_path = ct_path.replace(ct_root_path, output_root_path)
                seg_dir = os.path.split(seg_path)[0]
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)

                ct_image_array = np.reshape(ct_image_array, (1, height, width))
                seg_image = sitk.GetImageFromArray(ct_image_array)
                seg_image.CopyInformation(ct_image)
                sitk.WriteImage(seg_image, seg_path)
                logging.info(f'Save result to: {seg_path}')


def count_laa(data_dir_name):
    """
    计算肺气肿区（HU值小于-950）占肺实质区域的比例，即LAA
    :param data_dir_name:
    :return:
    """
    ct_root_path = "/data/zengnanrong/LUNG_SEG/train_valid"
    mask_root_path = "/data/zengnanrong/R231/"

    emphysema_pixel_num = 0
    lung_pixel_num = 0

    path = os.path.join(ct_root_path, data_dir_name)
    for root, dirs, files in os.walk(path):
        for item in files:
            if '.dcm' in item.lower():
                ct_path = os.path.join(root, item)
                mask_path = ct_path.replace(ct_root_path, mask_root_path)

                ct_image = sitk.ReadImage(ct_path)
                ct_image_array = np.squeeze(sitk.GetArrayFromImage(ct_image))
                mask_image = sitk.ReadImage(mask_path)
                mask_image_array = np.squeeze(sitk.GetArrayFromImage(mask_image))

                height = ct_image_array.shape[0]
                width = ct_image_array.shape[1]

                for h in range(height):
                    for w in range(width):
                        if mask_image_array[h][w] > 0:  # 肺区域
                            lung_pixel_num = lung_pixel_num + 1
                            if ct_image_array[h][w] <= -950:
                                emphysema_pixel_num = emphysema_pixel_num + 1

    laa = format(emphysema_pixel_num / lung_pixel_num, '.4f')
    print(f'{data_dir_name},{emphysema_pixel_num},{lung_pixel_num},{laa}')

def load_dicom(dicom_path):
    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_path)

    # 查看该文件夹下的序列数量

    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[0]代表的是第一个序列的ID

    # 如果不添加series_IDs[0]这个参数，则默认获取第一个序列的所有切片路径
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_path, series_IDs[0])

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    return image3D


def seg_nii(patient_dir_name):
    mask_root_path = '/data/zengnanrong/COPD_Biphasic_R231/stage_1/nii'
    patient_dir_path = os.path.join(mask_root_path, patient_dir_name)
    ct_dirs = get_ct_dirs(patient_dir_path)

    for dir in ct_dirs:
        mask_dir_path = os.path.join(patient_dir_path, dir)

        ct_path = mask_dir_path.replace('COPD_Biphasic_R231', 'COPD_Biphasic')
        ct_path = ct_path.replace('nii', 'DICOM')
        ct_image = load_dicom(ct_path)
        ct_image_array = sitk.GetArrayFromImage(ct_image)

        mask_nii_path = os.path.join(mask_dir_path, dir + '.nii')
        mask_image = sitk.ReadImage(mask_nii_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)

        depth, width, height = ct_image_array.shape
        for a in range(depth):
            for b in range(width):
                for c in range(height):
                    if mask_image_array[a][b][c] == 0:
                        ct_image_array[a][b][c] = 0

        seg_dir_path = mask_dir_path.replace('COPD_Biphasic_R231', 'COPD_Biphasic_seg')
        seg_nii_path = mask_nii_path.replace('COPD_Biphasic_R231', 'COPD_Biphasic_seg')
        if not os.path.exists(seg_dir_path):
            os.makedirs(seg_dir_path)
        seg_image = sitk.GetImageFromArray(ct_image_array)
        seg_image.CopyInformation(ct_image)
        sitk.WriteImage(seg_image, seg_nii_path)
        logging.info(f'Save result to: {seg_nii_path}')


if __name__ == "__main__":
    # input_root_path = "/data/zengnanrong/LUNG_SEG/train_valid"
    # input_root_path = "/data/zengnanrong/COPD_Biphasic/stage_3/DICOM"
    input_root_path = '/data/zengnanrong/COPD_Biphasic_R231/stage_1/nii'

    ct_dir = get_ct_dirs(input_root_path)

    pool = Pool(4)
    # pool.map(mask_dcms, ct_dir)
    # pool.map(segement_dcms, ct_dir)
    # pool.map(count_laa, ct_dir)
    # pool.map(mask_nii, ct_dir)
    pool.map(seg_nii, ct_dir)
    pool.close()
    pool.join()
