import edt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def get_next_click2D_torch(prev_seg, gt_semantic_seg):
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    for i in range(gt_semantic_seg.shape[0]):
        # fn_points shape: (N, 3) -> [channel, h, w]
        fn_points = torch.argwhere(fn_masks[i])
        fp_points = torch.argwhere(fp_masks[i])
        point = None
        
        if len(fn_points) > 0 and len(fp_points) > 0:
            if np.random.random() > 0.5:
                point = fn_points[np.random.randint(len(fn_points))]
                is_positive = True
            else:
                point = fp_points[np.random.randint(len(fp_points))]
                is_positive = False
        elif len(fn_points) > 0:
            point = fn_points[np.random.randint(len(fn_points))]
            is_positive = True
        elif len(fp_points) > 0:
            point = fp_points[np.random.randint(len(fp_points))]
            is_positive = False
        
        if point is None: 
            # Click đại một điểm nếu không có lỗi (thường là điểm âm)
            point = torch.tensor([0, np.random.randint(prev_seg.shape[2]), np.random.randint(prev_seg.shape[3])]).to(torch.int64)
            is_positive = False

        # Lấy (h, w), bỏ qua channel. Reshape cho SAM 2D: (1, 1, 2)
        bp = point[1:].clone().detach().reshape(1, 1, 2).to(pred_masks.device) 
        bl = torch.tensor([int(is_positive),]).reshape(1, 1).to(pred_masks.device) 

        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels

def get_next_click2D_torch_ritm(prev_seg, gt_semantic_seg):
    """ Chiến thuật click vào tâm vùng lỗi lớn nhất (RITM) cho 2D """
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    # Pad 2D: (left, right, top, bottom)
    fn_mask_single = F.pad(fn_masks.float(), (1,1,1,1), 'constant', value=0)[0,0]
    fp_mask_single = F.pad(fp_masks.float(), (1,1,1,1), 'constant', value=0)[0,0]
    
    # Tính Distance Transform 2D
    fn_mask_dt = torch.tensor(edt.edt(fn_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1]
    fp_mask_dt = torch.tensor(edt.edt(fp_mask_single.cpu().numpy(), black_border=True, parallel=4))[1:-1, 1:-1]
    
    fn_max_dist = torch.max(fn_mask_dt)
    fp_max_dist = torch.max(fp_mask_dt)
    
    is_positive = fn_max_dist > fp_max_dist
    dt = fn_mask_dt if is_positive else fp_mask_dt
    
    to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0)
    to_point_mask = to_point_mask[None, None] # Add batch & channel dims

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.nonzero(to_point_mask[i], as_tuple=False)
        if len(points) == 0: # Phòng trường hợp mask trống
            point = torch.tensor([0, 0, 0])
        else:
            point = points[np.random.randint(len(points))]
        
        # Kiểm tra lại loại point tại (channel 0, h, w)
        if fn_masks[i, 0, point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 2).to(prev_seg.device) 
        bl = torch.tensor([int(is_positive),]).reshape(1, 1).to(prev_seg.device)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels

def get_next_click2D_torch_2(prev_seg, gt_semantic_seg):
    """ Click ngẫu nhiên vào bất kỳ vùng lỗi nào (FN hoặc FP) """
    mask_threshold = 0.5
    batch_points = []
    batch_labels = []

    pred_masks = (prev_seg > mask_threshold)
    true_masks = (gt_semantic_seg > 0)
    fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    for i in range(gt_semantic_seg.shape[0]):
        points = torch.nonzero(to_point_mask[i], as_tuple=False)
        if len(points) == 0:
            point = torch.tensor([0, 0, 0])
        else:
            point = points[np.random.randint(len(points))]
            
        if fn_masks[i, 0, point[1], point[2]]:
            is_positive = True
        else:
            is_positive = False

        bp = point[1:].clone().detach().reshape(1, 1, 2).to(prev_seg.device) 
        bl = torch.tensor([int(is_positive),]).reshape(1, 1).to(prev_seg.device)
        batch_points.append(bp)
        batch_labels.append(bl)

    return batch_points, batch_labels

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_point(point, label, ax):
    # point định dạng (h, w)
    if label == 0:
        ax.add_patch(plt.Circle((point[1], point[0]), 2, color='red')) # Point âm
    else:
        ax.add_patch(plt.Circle((point[1], point[0]), 2, color='green')) # Point dương

if __name__ == "__main__":
    # Test thử với tensor 2D
    gt2D = torch.zeros((2, 1, 256, 256)).cuda()
    gt2D[:, :, 50:100, 50:100] = 1 # Tạo một vùng mask giả
    prev_masks = torch.zeros_like(gt2D).to(gt2D.device)
    
    batch_points, batch_labels = get_next_click2D_torch_ritm(prev_masks, gt2D)
    print(f"Points shape: {batch_points[0].shape}") # Kỳ vọng (1, 1, 2)
    print(f"Labels: {batch_labels}")