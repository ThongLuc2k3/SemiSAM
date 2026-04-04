
from segment_anything import sam_model_registry

import torch
import torch.nn as nn
import torch.nn.functional as F

# Chuyển đổi import sang click method
from utils.click_method import get_next_click2D_torch_ritm, get_next_click2D_torch_2

def compute_epistemic_uncertainty(all_preds):

    predictions = torch.stack(all_preds)
    ensemble = torch.mean(predictions, dim=0)

    uncertainty = []
    for pred in all_preds:
        exp_variance = torch.var(pred - ensemble)
        uncertainty.append(exp_variance)
    unc = torch.mean(torch.stack(uncertainty), dim=0)
    return unc

def finetune_model_predict2D_unc(img2D, gt2D, sam_model_tune, device='cuda', click_method='random', num_clicks=10, prev_masks=None):
    batch_size, ch, width, height = img2D.shape
    pred_batch = torch.zeros((batch_size, ch, width, height), device=device)
    unc_batch = torch.zeros((batch_size, ch, width, height), device=device)

    for b in range(batch_size):
        img2D_single = img2D[b:b + 1, ...]
        gt2D_single = gt2D[b:b + 1, ...]

        crop_size = 256 # Kích thước chuẩn cho SAM 2D thường là 1024, nhưng crop 256 để tiết kiệm mem

        if prev_masks is None:
            prev_masks = torch.zeros_like(img2D_single).to(device)
        
        # Nội suy 2D (Bilinear)
        low_res_masks = F.interpolate(prev_masks.float(), size=(crop_size // 4, crop_size // 4), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # Tự động chuyển 1 kênh sang 3 kênh cho SAM Encoder
            input_sam = img2D_single.repeat(1, 3, 1, 1) if img2D_single.shape[1] == 1 else img2D_single
            image_embedding = sam_model_tune.image_encoder(input_sam.to(device))

        all_preds = []
        all_preds.append(gt2D_single)
        
        current_prev_masks = prev_masks
        for num_click in range(num_clicks):
            with torch.no_grad():
                if num_click > 1:
                    click_method = get_next_click2D_torch_2
                
                batch_points, batch_labels = click_method(current_prev_masks.to(device), gt2D_single.to(device))

                points_input = torch.cat(batch_points, dim=0).to(device)
                labels_input = torch.cat(batch_labels, dim=0).to(device)

                sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                    points=[points_input, labels_input],
                    boxes=None,
                    masks=low_res_masks.to(device),
                )
                low_res_masks, _ = sam_model_tune.mask_decoder(
                    image_embeddings=image_embedding.to(device),
                    image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                # Nội suy 2D cho output
                current_prev_masks = F.interpolate(low_res_masks, size=img2D_single.shape[-2:], mode='bilinear', align_corners=False)
                all_preds.append(current_prev_masks)

        uncertainty = compute_epistemic_uncertainty(all_preds)
        medsam_seg_prob = current_prev_masks
        medsam_seg = (medsam_seg_prob > 0.5).to(torch.uint8)
        
        pred_batch[b:b + 1] = medsam_seg
        unc_batch[b:b + 1] = uncertainty

    return pred_batch, unc_batch

def finetune_model_predict2D_mask(img2D, gt2D, sam_model_tune, device='cuda'):
    batch_size, ch, width, height = img2D.shape
    pred_batch = torch.zeros((batch_size, ch, width, height), device=device)

    for b in range(batch_size):
        img2D_single = img2D[b:b + 1, ...]
        gt2D_single = gt2D[b:b + 1, ...]

        crop_size = 256
        low_res_gt = F.interpolate(gt2D_single.float(), size=(crop_size // 4, crop_size // 4), mode='bilinear', align_corners=False)

        with torch.no_grad():
            input_sam = img2D_single.repeat(1, 3, 1, 1) if img2D_single.shape[1] == 1 else img2D_single
            image_embedding = sam_model_tune.image_encoder(input_sam.to(device))
            
            sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
                points=None,
                boxes=None,
                masks=low_res_gt.to(device)
            )
            low_res_masks, _ = sam_model_tune.mask_decoder(
                image_embeddings=image_embedding.to(device),
                image_pe=sam_model_tune.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            masks = F.interpolate(low_res_masks, size=img2D_single.shape[-2:], mode='bilinear', align_corners=False)
            medsam_seg = (masks > 0.5).to(torch.uint8)

        pred_batch[b:b + 1] = medsam_seg

    return pred_batch

def semisam_branch(image_batch, mask, generalist='SAM', prompt='mask', device='cuda'):
    """
    Cổng kết nối 2D. 
    generalist: 'SAM' hoặc 'MedSAM'
    """
    if generalist == 'SAM':
        checkpoint_path = '../ckpt/sam_vit_b_01ec14.pth'
        model_type = 'vit_b'
    elif generalist == 'MedSAM':
        checkpoint_path = '../ckpt/medsam_vit_b.pth'
        model_type = 'vit_b'
    else:
        # Fallback cho các trường hợp khác
        checkpoint_path = '../ckpt/sam_vit_b_01ec14.pth'
        model_type = 'vit_b'

    # Khởi tạo SAM 2D
    sam_model_tune = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
    sam_model_tune.eval()

    unc = []
    if prompt == 'point':
        samseg_mask = finetune_model_predict2D_unc( # Sử dụng bản unc nhưng lấy mask
            image_batch, mask, sam_model_tune, device=device,
            click_method=get_next_click2D_torch_ritm, num_clicks=10)[0]
    elif prompt == 'mask':
        samseg_mask = finetune_model_predict2D_mask(
            image_batch, mask, sam_model_tune, device=device) 
    elif prompt == 'unc':
        samseg_mask, unc = finetune_model_predict2D_unc(
            image_batch, mask, sam_model_tune, device=device,
            click_method=get_next_click2D_torch_ritm, num_clicks=10) 
    
    return samseg_mask, unc