import time
import uuid
from pathlib import Path

import pyautogui
import win32api
import win32con
import cv2
import keyboard
from ultralytics.utils.plotting import Annotator

from utils.general import non_max_suppression, scale_boxes
import torch
from utils.dataloaders import LoadScreenshots
from models.common import DetectMultiBackend
from utils.plots import Colors
# v1.0
# 是否保存图像用于后续训练
save_image = False
# 图像保存路径
image_save_path = 'cache'
# 使用的权重文件，需要自行训练或者使用官方提供的版本
weight_path = 'runs/train/exp6/weights/best.pt'
# 数据集配置文件
cfg = 'data/ssjj.yaml'
# 是否显示实时目标检测画面，使用双屏效果最佳
screen_show = False
# 偏移量系数，决定鼠标移动到目标物体一次的移动量
shifting_α = 1000
# 置信度阈值
conf_thres = 0.4

dataset = LoadScreenshots('screen 1 288 0 1344 1080')
model = DetectMultiBackend(weight_path, device=torch.device('cuda'), dnn=False, data=cfg, fp16=False)
names = model.names
colors = Colors()


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calculate_bbox_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_offset(to_x, to_y):
    # 数值越大x,y偏移量越多
    fov_horizontal = shifting_α
    fov_vertical = fov_horizontal * 9 / 13.8
    target_x = to_x
    target_y = to_y
    cross_hair_x = 672.0
    cross_hair_y = 540.0
    x_move = (target_x - cross_hair_x) / 1920.0 * fov_horizontal
    y_move = (target_y - cross_hair_y) / 1080.0 * fov_vertical
    return int(x_move), int(y_move)


# 键盘事件监听


running = False


def on_f10():
    global running
    running = True


def on_f12():
    global running
    running = False


keyboard.add_hotkey('F10', on_f10)
keyboard.add_hotkey('F12', on_f12)
while True:
    if running:
        for path, im, im0s, vid_cap, s in dataset:
            if not running:
                break
            if save_image:
                cv2.imwrite(str(Path(image_save_path) / uuid.uuid1().hex) + '.jpg', im0s)
            # 索引1为被缩小后的图，640
            # 索引2为原始图
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, 0.45, 0, False, max_det=10)
            for i, det in enumerate(pred):  # per image
                if len(det):
                    # 将候选框调整至原始大小
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                # 先计算操作再画图，不然有延迟
                if len(det):
                    positions = []

                    for *xyxy, conf, cls in reversed(det):
                        # 计算候选框中心点
                        bbox_center_x, bbox_center_y = calculate_bbox_center(*xyxy)
                        # 存储所有中心点和候选框中心点
                        positions.append((672, 540, bbox_center_x.item(), bbox_center_y.item()))
                    # 找出最近的候选框中心点
                    position = min(positions, key=lambda item: euclidean_distance(*item))
                    # 计算鼠标偏移距离
                    x, y = calculate_offset(position[2], position[3])
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y)
                    pyautogui.click()
                if screen_show:
                    # 画面渲染
                    im0 = im0s.copy()
                    annotator = Annotator(im0, line_width=3, example=str(names))
                    if len(det):
                        # 绘画窗口
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f"{names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))

                            bbox_center_x, bbox_center_y = calculate_bbox_center(*xyxy)
                            cv2.circle(annotator.im, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                            cv2.line(annotator.im, (672, 540), (int(bbox_center_x), int(bbox_center_y)), (0, 0, 255, 2))
                    im0 = annotator.result()
                    cv2.imshow('frame', cv2.resize(im0, (int(im0.shape[1] * 0.7), int(im0.shape[0] * 0.7))))
                    c = cv2.waitKey(1)  # 1 millisecond
                    if c == 27:
                        break
            time.sleep(0.004)
    time.sleep(1)