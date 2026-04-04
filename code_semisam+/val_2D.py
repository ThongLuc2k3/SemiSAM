import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
from PIL import Image

def test_single_case(net, image, stride_xy, patch_size, num_classes=1):
    """ Xử lý test cho 1 ảnh 2D """
    w, h = image.shape

    # Nếu kích thước ảnh nhỏ hơn patch_size thì thực hiện padding
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2

    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad)], mode='constant', constant_values=0)
    
    ww, hh = image.shape

    # Tính số bước nhảy (sliding window)
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            
            # Cắt patch 2D
            test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1]]
            # Expand dims thành (1, 1, H, W) để đưa vào model
            test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
            test_patch = torch.from_numpy(test_patch).cuda()

            with torch.no_grad():
                y1 = net(test_patch)
                y = torch.softmax(y1, dim=1)
            
            y = y.cpu().data.numpy()
            y = y[0, :, :, :] # Lấy (num_classes, H, W)

            # Cộng dồn kết quả vào score_map
            score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1]] \
                += y
            cnt[xs:xs + patch_size[0], ys:ys + patch_size[1]] += 1

    # Chia trung bình các vùng chồng lấp
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    # Loại bỏ padding nếu có
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h]

    return label_map

def cal_metric(gt, pred):
    """ Tính toán Dice và HD95 """
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    elif pred.sum() == 0 and gt.sum() == 0:
        return np.array([1.0, 0.0]) # Cả hai đều trống thì coi như khớp hoàn hảo
    else:
        return np.array([0.0, 50.0]) # Sai lệch hoàn toàn (HD95 gán giá trị phạt lớn)

def test_all_case(net, base_dir, test_list="val.txt", num_classes=2, patch_size=(256, 256), stride_xy=32):
    """ Loop qua danh sách val để tính metric trung bình """
    with open(os.path.join(base_dir, test_list), 'r') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    
    total_metric = np.zeros((num_classes - 1, 2))
    print(f"Validation begin: {len(image_names)} cases")

    for name in tqdm(image_names):
        # Đường dẫn tới ảnh và mask của BTXRD
        img_path = os.path.join(base_dir, "images", name + ".png") # Giả định định dạng .png
        lab_path = os.path.join(base_dir, "mask", name + ".png")

        # Đọc ảnh và normalize sơ bộ (chia 255)
        image = np.array(Image.open(img_path).convert('L')) / 255.0
        label = np.array(Image.open(lab_path).convert('L'))
        # Đưa label về dạng 0, 1 nếu cần
        label[label > 0] = 1

        prediction = test_single_case(net, image, stride_xy, patch_size, num_classes=num_classes)

        # Tính metric cho từng class (bỏ qua background class 0)
        for i in range(1, num_classes):
            total_metric[i - 1, :] += cal_metric(label == i, prediction == i)

    print("Validation end")
    return total_metric / len(image_names)