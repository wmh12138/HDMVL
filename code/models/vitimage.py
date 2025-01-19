import torch
import torch.nn as nn
import torchvision
import os
import clip

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args

        current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("当前设备: ", current_device)
        model, preprocess = clip.load('ViT-B/32')
        model = model.encode_image
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
