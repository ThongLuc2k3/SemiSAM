import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
# Đổi từ brats2019 sang BTXRD (phù hợp với data X-quang của bạn)
from dataloaders.BTXRD import (BTXRD, RandomRotFlip, ToTensor, TwoStreamBatchSampler)
from networks.net_factory import net_factory # Sử dụng factory cho 2D
from utils import losses, metrics, ramps
from val_2D import test_all_case # Chuyển sang file val 2D

from semisam_plus import semisam_branch

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BTXRD', help='Path to BTXRD data')
parser.add_argument('--exp', type=str,
                    default='BTXRD/SemiSAM_MT', help='experiment_name')
parser.add_argument('--prompt', type=str,
                    default='unc')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name (e.g. unet, enet)')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input (2D)')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data count')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2

    def create_model(ema=False):
        # Chuyển sang net_factory 2D
        net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    from semisam_plus import SamBuildArgs
    from segment_anything import sam_model_registry

    print("------- ĐANG NẠP MÔ HÌNH SAM -------")
    build_args = SamBuildArgs(
       image_size=256, 
       checkpoint='./ckpt/sam-med2d_b.pth', 
       encoder_adapter=True
    )

    sam_model_obj = sam_model_registry['vit_b'](build_args).cuda()
    sam_model_obj.eval()

    # Sử dụng bộ dataset BTXRD cho ảnh X-quang
    db_train = BTXRD(base_dir=train_data_path,
                         split='train',
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             ToTensor(),
                         ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    # Giả định tổng số data trong list là lớn hơn labeled_num, ví dụ 100
    unlabeled_idxs = list(range(args.labeled_num, len(db_train))) 
    
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:] - ema_output_soft)**2)

            # semisam_branch: Bạn cần đảm bảo generalist chuyển sang 'SAM' hoặc 'MedSAM' (2D)
            # Truyền volume_batch (B, 1, H, W) và output map
            samseg_mask, uncsam = semisam_branch(volume_batch, outputs_soft[:,1:2,:,:],  sam_model_tune=sam_model_obj, prompt=args.prompt)
            
            samseg_soft = torch.cat((1 - samseg_mask, samseg_mask), dim=1)

            if args.prompt == 'unc':
                sam_consistency_dist = (outputs_soft[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2
                sam_consistency = torch.mean(
                    sam_consistency_dist * uncsam) / (torch.mean(uncsam) + 1e-8) + torch.mean(uncsam)
            else:
                sam_consistency = torch.mean(
                    (outputs_soft[args.labeled_bs:] - samseg_soft[args.labeled_bs:])**2)

            consistency_weight_sam = get_current_consistency_weight((max_iterations - iter_num)//150)
            sam_con_loss = 0.1 * consistency_weight_sam * sam_consistency

            loss = supervised_loss + consistency_weight * consistency_loss + sam_con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # THAY iterator.set_description BẰNG LỆNH PRINT NÀY:
            print(f"Iter {iter_num}/{max_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Dice: {loss_dice.item():.4f} | "
                  f"CE: {loss_ce.item():.4f} | "
                  f"SAM: {sam_con_loss.item():.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Logging
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            if iter_num % 20 == 0:
                # Visualization sửa lại cho 2D (hiển thị trực tiếp batch)
                image = volume_batch[0, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)

                image_pred = outputs_soft[0, 1:2, :, :]
                writer.add_image('train/Predicted_label', image_pred, iter_num)

                image_gt = label_batch[0, :, :].unsqueeze(0)
                writer.add_image('train/Groundtruth_label', image_gt, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                # test_all_case cho 2D: patch_size 2D, không có stride_z
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=32)
                
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', avg_metric[0, 0], iter_num)
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)