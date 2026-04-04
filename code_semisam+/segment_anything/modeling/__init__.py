# Cấu trúc dành cho 2D (Sử dụng cho X-quang xương)
from .sam_model import Sam
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

# Cấu trúc dành cho 3D (Nếu bạn vẫn để các file này trong folder)
from .sam3D import Sam3D
from .image_encoder3D import ImageEncoderViT3D
from .mask_decoder3D import MaskDecoder3D, TwoWayTransformer3D
from .prompt_encoder3D import PromptEncoder3D