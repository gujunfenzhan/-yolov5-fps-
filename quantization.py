import argparse

import onnx
from onnxruntime.quantization import QuantType, quantize_static,  CalibrationDataReader
import onnxruntime
import cv2
import os
import numpy as np


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = cv2.imread(image_filepath)
        h, w, c = pillow_img.shape
        scale = 640 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = cv2.resize(pillow_img, (new_w, new_h))
        # 中心裁剪出 target_size x target_size 的图像
        start_x = (new_w - width) // 2
        start_y = (new_h - height) // 2
        cropped_image = resized_image[start_y:start_y + height, start_x:start_x + width]
        # 将图像格式从 [高, 宽, 通道数] 转换为 [通道数, 高, 宽]
        transformed_image = np.transpose(cropped_image, (2, 0, 1))
        transformed_image = transformed_image / 255
        unconcatenated_batch_data.append(transformed_image)

    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    batch_data = batch_data.reshape((batch_data.shape[0], 1, *batch_data.shape[1:]))
    return batch_data.astype(np.float32)


class yolov5_cal_data_reader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


parser = argparse.ArgumentParser()
# 需要量化的文件
parser.add_argument('--source', type=str, required=True)
# 剪枝后的权重文件
parser.add_argument('--target', type=str, default=None, required=False)
# 数据集文件夹
parser.add_argument('--datasets', type=str, default="datasets/ssjj")

args = parser.parse_args()
# 打印选择参数
print("#########################################")
for item, value in args.__dict__.items():
    print(item, value)
print("#########################################")
source_path = args.source
if args.target is None:
    target_path = args.source.replace('.pth', '_quantization.pth')
else:
    target_path = args.target

model_path = source_path
outout_model_quant = target_path

dr = yolov5_cal_data_reader(args.datasets, model_path)

quantize_static(
    model_path,
    outout_model_quant,
    calibration_data_reader=dr,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    # 忽略的节点，部分节点使用量化的效果并不好，就不进行量化处理
    nodes_to_exclude=[
        "/model.24/Concat_3", "/model.24/Reshape_1", "/model.24/Reshape_3",
        "/model.24/Reshape_5", "/model.24/Concat", "/model.24/Concat_1",
        "/model.24/Concat_2", "/model.24/Split_2", "/model.24/Mul_1",
        "/model.24/Add", "/model.24/Mul_5", "/model.24/Add_1",
        "/model.24/Mul_9", "/model.24/Split_1", "/model.24/Add_2",
        "/model.23/m/m.0/cv2/act/Sigmoid", "/model.23/m/m.0/cv1/act/Sigmoid"],
    per_channel=False,
    reduce_range=False,
)
