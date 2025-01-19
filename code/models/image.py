import torch
import torch.nn as nn
import torchvision
from vision_mamba import Vim

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        model = Vim(
            dim=256,  # Dimension of the transformer model
            dt_rank=32,  # Rank of the dynamic routing matrix
            dim_inner=256,  # Inner dimension of the transformer model
            d_state=256,  # Dimension of the state vector
            num_classes=args.img_hidden_sz,  # Number of output classes
            image_size=args.FINE_SIZE,  # Size of the input image
            patch_size=32,  # Size of each image patch
            channels=3,  # Number of input channels
            dropout=0.1,  # Dropout rate
            depth=12,  # Depth of the transformer model
        )
        #modules = list(model.children())
        #变量前加一个星号*，目的是将该list变量拆解开多个独立的参数，传入函数中
        self.model = model

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
       out = self.model(x)
       #print('model out',out.shape)
       out = self.pool(out)
       #print('pool out',out.shape)
       out = torch.flatten(out, start_dim=2)
       #print('flatten out',out.shape)
       out = out.transpose(1, 2).contiguous()
       #print('transpose out',out.shape)
       return out  # BxNx2048
