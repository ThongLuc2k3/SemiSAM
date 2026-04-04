from networks.unet import UNet
from networks.attention_unet import Attention_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.nnunet import initialize_network_2d

def net_factory(net_type="unet", in_chns=1, class_num=2):
    """
    Factory function to create 2D segmentation networks.
    """
    if net_type == "unet":
        # Giả định UNet 2D nhận tham số n_classes và in_channels
        net = UNet(in_channels=in_chns, n_classes=class_num).cuda()
        
    elif net_type == "attention_unet":
        # Attention UNet cho ảnh 2D (rất tốt cho X-quang)
        net = Attention_UNet(in_channels=in_chns, n_classes=class_num).cuda()
        
    elif net_type == "enet":
        # ENet nhẹ hơn, phù hợp nếu cần tốc độ
        net = ENet(in_channels=in_chns, n_classes=class_num).cuda()
        
    elif net_type == "pnet":
        # Một biến thể thường dùng trong Semi-supervised
        net = PNet2D(in_channels=in_chns, n_classes=class_num).cuda()
        
    elif net_type == "nnUNet":
        # Phiên bản 2D của nnUNet
        net = initialize_network_2d(num_classes=class_num).cuda()
        
    else:
        net = None
        print(f"Error: Model type '{net_type}' is not supported in 2D factory.")
        
    return net