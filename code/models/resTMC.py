import torch
import torch.nn as nn
from models.resimage import ImageEncoder
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# loss function
# def HD(alpha, c):
#     a = 1.8
#     b = a/(a-1)
#     beta = torch.ones((1, c)).cuda()
#
#     F1_1 = torch.sum(torch.lgamma(a * alpha + beta), dim=1, keepdim=True) - torch.lgamma(torch.sum((a * alpha + beta), dim=1, keepdim=True))
#     F2_1 = torch.sum(torch.lgamma((b + 1) * beta), dim=1, keepdim=True) - torch.lgamma(torch.sum(((b + 1) * beta), dim=1, keepdim=True))
#     F3_1 = torch.sum(torch.lgamma(alpha + 2 * beta), dim=1, keepdim=True) - torch.lgamma(torch.sum((alpha + 2 * beta), dim=1, keepdim=True))
#
#     hd1 = 1/a * F1_1 + 1/b * F2_1 - F3_1
#
#     a = b
#     b = a/(a-1)
#     F1_2 = torch.sum(torch.lgamma(a * alpha + beta), dim=1, keepdim=True) - torch.lgamma(torch.sum((a * alpha + beta), dim=1, keepdim=True))
#     F2_2 = torch.sum(torch.lgamma((b + 1) * beta), dim=1, keepdim=True) - torch.lgamma(torch.sum(((b + 1) *  beta), dim=1, keepdim=True))
#     F3_2 = torch.sum(torch.lgamma(alpha + 2 * beta), dim=1, keepdim=True) - torch.lgamma(torch.sum((alpha + 2 * beta), dim=1, keepdim=True))
#
#     hd2 = 1/a * F1_2 + 1/b * F2_2 - F3_2
#
#     hd = 1/2 * (hd1+hd2)
#     return hd
# loss function
def JE(alpha, c):
    beta = torch.ones((1, c)).cuda()
    q = F.softmax(alpha, dim=1)
    p = F.softmax(beta, dim=1)
    kl_pq = F.kl_div(q.log(), p, reduction='batchmean')
    kl_qp = F.kl_div(p.log(), q, reduction='batchmean')

    loss_Je = kl_pq + kl_qp
    return loss_Je

def ce_loss(p, alpha, c, global_step, annealing_step):
    #print(alpha)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    #将标签转成one-hot编码
    label = F.one_hot(p, num_classes=c)
    #print("Label shape:", label.shape)
    #print("S shape:", S.shape)
    #print("Alpha shape:", alpha.shape)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * JE(alp, c)
    #print('A:{},HD:{}'.format(A,B))
    #print('A+B:{}'.format(torch.mean((A + B))))
    return torch.mean((A + B))

# def ce_loss(p, alpha, c, global_step, annealing_step):
#     S = torch.sum(alpha, dim=1, keepdim=True)
#     E = alpha - 1
#     #将标签转成one-hot编码
#     label = F.one_hot(p, num_classes=c)
#     #print("Label shape:", label.shape)
#     #print("S shape:", S.shape)
#     #print("Alpha shape:", alpha.shape)
#     A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
#
#     annealing_coef = min(1, global_step / annealing_step)
#     alp = E * (1 - label) + 1
#     B = annealing_coef * HD(alp, c)
#     return torch.mean((A + B))


class TMC(nn.Module):
    def __init__(self, args):
        super(TMC, self).__init__()
        self.args = args
        #获得模型初始化参数，其中池化层是自定义的
        self.rgbenc = ImageEncoder(args)
        self.depthenc = ImageEncoder(args)
        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds
        #nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
        self.clf_depth = nn.ModuleList()
        self.clf_rgb = nn.ModuleList()
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))

        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            #计算所有参数的和然后保留这个维度
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            #将单个维度扩大成更大维度，返回一个新的tensor
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # b^0 @ b^(0+1)   torch.bmm计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m)
        #也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        # b^0 * u^1
        #torch.mul是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)  #计算bb内所有参数的和
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)  #求bb内对角线附件的元素，并将最后一个维度求和
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.args.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, rgb, depth):
        #首先输入depth数据到resnet18分类网络中，然后将分类结果作为预训练的结果扁平化处理
        depth = self.depthenc(depth)
        depth = torch.flatten(depth, start_dim=1)
        #首先输入rgb数据到resnet18分类网络中，然后将分类结果作为预训练的结果扁平化处理
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)

        #对通过模型取得分类结果
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        #Softplus函数可以看作是ReLU激活函数的平滑代替。
        depth_evidence, rgb_evidence = F.softplus(depth_out), F.softplus(rgb_out)
        depth_alpha, rgb_alpha = depth_evidence+1, rgb_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(depth_alpha, rgb_alpha)
        return depth_alpha, rgb_alpha, depth_rgb_alpha


class ETMC(TMC):
    def __init__(self, args):
        super(ETMC, self).__init__(args)
        #这一部分需要研究一下
        last_size = args.img_hidden_sz * args.num_image_embeds + args.img_hidden_sz * args.num_image_embeds
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden
        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, rgb, depth):
       # print("depth shape:", depth.shape)
        depth = self.depthenc(depth)
       # print("depthenc shape:", depth.shape)
        depth = torch.flatten(depth, start_dim=1)
        #print("flatted depth shape:", depth.shape)
        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)
        depth_out = depth
        for layer in self.clf_depth:
            depth_out = layer(depth_out)
        #print("depth_out shape:", depth_out.shape)
        rgb_out = rgb
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)

        #按列拼接数据
        pseudo_out = torch.cat([rgb, depth], -1)
        for layer in self.clf:
            pseudo_out = layer(pseudo_out)

        depth_evidence, rgb_evidence, pseudo_evidence = F.softplus(depth_out), F.softplus(rgb_out), F.softplus(pseudo_out)
        #print("depth_evidence shape:", depth_evidence.shape)
        depth_alpha, rgb_alpha, pseudo_alpha = depth_evidence+1, rgb_evidence+1, pseudo_evidence+1
        depth_rgb_alpha = self.DS_Combin_two(self.DS_Combin_two(depth_alpha, rgb_alpha), pseudo_alpha)
        return depth_alpha, rgb_alpha, pseudo_alpha, depth_rgb_alpha
