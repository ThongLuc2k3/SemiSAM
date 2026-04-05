import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
# from tqdm import tqdm  # Loại bỏ tqdm theo yêu cầu để tránh spam log
from PIL import Image

def test_single_case(net, image, stride_xy, patch_size, num_classes=2):
    """ Xử lý test cho 1 ảnh 2D bằng Sliding Window """
    image = np.array(image)
    w, h = image.shape

    # Padding nếu ảnh nhỏ hơn patch_size
    add_pad = False
    if w < patch_size[0] or h < patch_size[1]:
        w_pad = max(0, patch_size[0] - w)
        h_pad = max(0, patch_size[1] - h)
        add_pad = True
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad)], mode='constant', constant_values=0)
    else:
        wl_pad, hl_pad = 0, 0

    ww, hh = image.shape

    # Tính số bước nhảy (sliding window)
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    
    score_map = np.zeros((num_classes, ww, hh)).astype(np.float32)
    cnt = np.zeros((ww, hh)).astype(np.float32)

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
                y_logit = net(test_patch)
                y_prob = torch.softmax(y_logit, dim=1)
            
            y_prob = y_prob.cpu().data.numpy()[0]
            score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1]] += y_prob
            cnt[xs:xs + patch_size[0], ys:ys + patch_size[1]] += 1

    # Chia trung bình các vùng chồng lấp
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    # Loại bỏ padding nếu có
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h]

    return label_map

def calculate_metric_per_case(pred, gt):
    """ Tính Dice và HD95 an toàn """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0 
    else:
        return 0.0, 50.0

def test_all_case(net, base_dir, test_list="val.txt", num_classes=2, patch_size=(256, 256), stride_xy=32):
    """ Đánh giá tập Validation với thông báo tiến độ tối giản """
    list_path = os.path.join(base_dir, test_list)
    if not os.path.exists(list_path):
        print(f"⚠️ Không tìm thấy file list tại: {list_path}")
        return np.zeros((num_classes - 1, 2))

    with open(list_path, 'r') as f:
        image_names = [line.strip() for line in f.readlines() if line.strip()]
    
    total_samples = len(image_names)
    total_metric = np.zeros((num_classes - 1, 2))
    
    # Thiết lập khoảng in tiến độ (in khoảng 5 lần)
    num_updates = 5
    update_interval = max(1, total_samples // num_updates)

    print(f"--- Bắt đầu Validation ({total_samples} mẫu) ---")

    net.eval()
    for idx, name in enumerate(image_names):
        # Log tiến độ tại các khoảng nhất định
        if (idx + 1) % update_interval == 0 or (idx + 1) == total_samples:
            progress = ((idx + 1) / total_samples) * 100
            print(f"  > Progress: {idx + 1}/{total_samples} ({progress:.0f}%) -- Đang xử lý: {name}")

        img_path = os.path.join(base_dir, "images", name + ".png")
        lab_path = os.path.join(base_dir, "masks", name + ".png")
        if not os.path.exists(lab_path):
            lab_path = os.path.join(base_dir, "mask", name + ".png")

        try:
            image = np.array(Image.open(img_path).convert('L')) / 255.0
            label = np.array(Image.open(lab_path).convert('L'))
            label[label > 0] = 1

            prediction = test_single_case(net, image, stride_xy, patch_size, num_classes=num_classes)

            for i in range(1, num_classes):
                dice, hd = calculate_metric_per_case(prediction == i, label == i)
                total_metric[i - 1, 0] += dice
                total_metric[i - 1, 1] += hd
        except Exception as e:
            print(f"  ❌ Lỗi tại mẫu {name}: {e}")
            continue

    avg_metric = total_metric / total_samples
    print(f"✅ Hoàn thành Val | Mean Dice: {avg_metric[0, 0]:.4f} | Mean HD95: {avg_metric[0, 1]:.4f}")
    return avg_metric