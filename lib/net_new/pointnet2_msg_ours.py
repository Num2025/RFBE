import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from torch.nn.functional import grid_sample
from lib.net.utils import k_nearest_neighbor
from lib.utils.sample2grid import sample2grid_F, sample2GaussianGrid_F, sample2BilinearGrid_F
from lib.net.self_attention import PointContext3D
from lib.net.clfm import CLFM
from lib.net.clfm import FusionAwareInterpii

BatchNorm2d = nn.BatchNorm2d
from lib.net.fft import FACMA


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def unsorted_segment_max_pytorch(data, segment_ids):
    num_segments = 2  # 计算段的数量
    result = torch.zeros(num_segments, *data.shape[1:], dtype=data.dtype, device=data.device)
    data = data.unsqueeze(-1)
    # 对每个段进行迭代
    for segment in range(num_segments):
        # 选择属于当前段的数据
        mask = (segment_ids == segment).unsqueeze(-1).expand_as(data)
        segment_data = data[mask].view(-1, data.size(-1))
        if segment_data.numel() == 0:
            continue
        # 计算当前段的最大值
        max_values, _ = torch.max(segment_data, dim=0)

        # 将最大值分配到结果张量中
        result[segment] = max_values

    return result


def unsorted_segment_sum_pytorch(data, segment_ids):
    num_segments = 2
    result = torch.zeros(num_segments, *data.shape[1:], dtype=data.dtype, device=data.device)

    for segment in range(num_segments):
        mask = (segment_ids == segment)
        # segment_data = data[mask].view(-1, data.size(-1))
        # data = data.unsqueeze(-1)
        # print("data:",data.size())
        # print("mask:",mask.size())
        segment_sum = (data * mask.float()).sum(dim=0)
        result[segment] = segment_sum

    return result


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndimension() - 1)
        # Generate random tensor
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(attn_output + x)
        fc_output = self.fc(x)
        x = self.norm2(fc_output + x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(2).permute(2, 0, 1)


class Unflatten(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(Unflatten, self).__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

    def forward(self, x):
        return x.permute(1, 2, 0).view(-1, self.embed_dim, self.height, self.width)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = ConvBN(inplanes, outplanes, 3, 2, 1)
        # self.relu = nn.ReLU6()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes, 2 * stride)
        # self.bn2 = BatchNorm2d(outplanes)
        # self.self_attention = nn.MultiheadAttention(embed_dim=outplanes,num_heads=8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        # # Flatten the spatial dimensions (H and W) into a single dimension for attention mechanism
        # batch_size, channels, height, width = out.size()
        # out = out.view(batch_size, channels, height * width).permute(0, 2, 1)  # (batch_size, H*W, channels)
        #
        # # Apply self-attention
        # attn_output, _ = self.self_attention(out, out, out)
        # attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, height,
        #                                                 width)  # (batch_size, channels, H, W)

        return out
        # out = self.conv1(x)
        # out = self.relu(out)
        # return out


class BasicBlock2(nn.Module):
    def __init__(self, inplanes, outplanes, mlp_ratio=3, drp_path=0.):
        super().__init__()
        self.conv1 = ConvBN(inplanes, outplanes, 3, 2, 1)
        self.star = Block(outplanes, mlp_ratio, drp_path)

    def forward(self, x):
        out = self.conv1(x)
        out = self.star(out)
        return out


class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


class Fusion_Cross_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Fusion_Cross_Conv, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = conv3x3(inplanes, outplanes, stride=1)
        self.bn1 = BatchNorm2d(outplanes)

    def forward(self, point_features, img_features):
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


class P2IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION PI2 ATTENTION#########')
        super(P2IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.ic // 4
        # self.q_embedding = nn.Linear(self.pc, rc)
        # self.k_embedding = nn.Linear(self.ic, rc)
        # self.v_embedding = nn.Linear(self.ic, rc)
        # self.output_embedding = nn.Linear(rc, self.pc)
        # self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv1d(self.pc, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)
        self.fc4 = nn.Linear(3, rc)
        # self.conv1 = nn.Sequential(nn.Conv1d(self.pc, rc, 1),  #####
        #                            nn.BatchNorm1d(rc),  ####
        #                            nn.ReLU())
        # self.fcK = nn.Linear(self.pc, rc)  # K
        # self.fcV = nn.Linear(self.pc, rc)  # V
        # self.fcQ = nn.Linear(self.ic, rc)  # Q
        # self.fc3 = nn.Linear(rc, 1)
        # self.fcout = nn.Linear(rc, self.pc)

    def forward(self, img_feas, point_feas,xyz):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        # query = self.q_embedding(point_feas_f)
        # key = self.k_embedding(img_feas_f)
        # value = self.v_embedding(img_feas_f)
        # # 计算点云特征和图像特征之间的注意力权重
        # attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        #     torch.tensor(query.size(-1), dtype=torch.float32))
        # attention_weights = self.softmax(attention_scores)  # (B, N, N)
        #
        # # 使用注意力权重加权图像特征
        # weighted_img_features = torch.matmul(attention_weights, value)  # (B, N, D)
        #
        # # 输出融合后的图像特征
        # out = self.output_embedding(weighted_img_features)  # (B, N, C_i)
        # out = out.transpose(1, 2)
        # print(img_feas)
        ################
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        rz = self.fc4(xyz)
        att = F.sigmoid(self.fc3(F.tanh(rz*ri + rp)))  # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1)  # B1N
        # # print(img_feas.size(), att.size())
        #
        point_feas_new = self.conv1(point_feas)
        out = point_feas_new * att
        # print("out.size:",out.size())
        ################
        # K = self.fcK(point_feas_f)
        # V = self.fcV(point_feas_f)
        # Q = self.fcQ(img_feas_f)
        # rc = self.ic // 4
        # att_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / K.size(-1) ** 0.5, dim=-1)
        # att_output = torch.matmul(att_scores, V)  # pc,rc
        # point_feas = self.conv1(point_feas)
        # point_feas_ff = point_feas.transpose(1, 2).contiguous().view(-1, rc)
        # fc_out = self.fcout(point_feas_ff + att_output)
        # out = fc_out.view(batch, -1, self.pc)
        # out = out.permute(0, 2, 1)
        return out


class Crossalign(nn.Module):
    def __init__(self, image_c, Lidar_c, qkv_c=128):
        super().__init__()
        self.ic, self.pc, self.qkv = image_c, Lidar_c, qkv_c
        self.q_embedding = nn.Linear(self.pc, self.qkv)  # lidar->q
        self.k_embedding = nn.Linear(self.ic, self.qkv)  # k
        self.v_embedding = nn.Linear(self.ic, self.qkv)  # v
        self.wsoftmax = nn.Softmax(dim=-1)
        # self.i2p = nn.Linear(self.ic,self.pc)
        self.output = nn.Linear(self.qkv, self.ic)
        self.bni=nn.BatchNorm1d(self.ic)
        self.bnp=nn.BatchNorm1d(self.pc)
        self.fc1=nn.Linear(self.ic,self.ic//4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.ic//4,self.ic)
        # self.dropout= nn.Dropout(0.7)

    def forward(self, img_feas, point_feas, li_index):
        # img_feas(B,C,H,W)
        # point_feas(B,C,N)
        # li_index(B,N)
        # print("img_feas:", img_feas.size())
        # print("point_feas:", point_feas.size())
        # print("li_index:",li_index.size())


        img_feas = img_feas.permute(0, 2, 3, 1)  # B,H,W,C
        flat_img = img_feas.reshape(img_feas.size(0), -1, img_feas.size(3))
        li_index_expanded = li_index.unsqueeze(-1).expand(-1, -1, self.ic).long()
        # print("li_index:",li_index_expanded.size())
        # print("flat_img:", flat_img.size())
        img_feature_fusion = torch.gather(flat_img, 1, li_index_expanded)  # B N C


        img_feature_fusion = self.bni(img_feature_fusion.permute(0, 2, 1)).permute(0,2,1)

        # print("point_feas:",point_feas.size()) #[2,96,4096] B C N
        # print("img_feature_fusion:",img_feature_fusion.size())#B N C
        # img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        point_feas_norm = self.bnp(point_feas)
        point_feas_f = point_feas_norm.transpose(1, 2).contiguous()  # BCN->BNC->(BN)C'

        q = self.q_embedding(point_feas_f)
        k = self.k_embedding(img_feature_fusion)
        v = self.v_embedding(img_feature_fusion)
        # print("q:",q.size()) 2,4096,128
        # print("k:", k.size())B,N,C
        # print("v:", v.size())B,N,C
        affinity = torch.einsum('bnc,bnc->bn', q, k) / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
        # print("affinity:", affinity.size())  # B,N
        # maxaffinity = unsorted_segment_max_pytorch(affinity, li_index)
        # # print(maxaffinity.size())
        # # print(li_index.size())
        # maxaffinity = torch.gather(maxaffinity, 1, li_index.long())
        # e_affinity = torch.exp(affinity - maxaffinity)
        # # print(e_affinity.size())
        # e_affinity_sum = unsorted_segment_sum_pytorch(e_affinity, li_index)
        # e_affinity_sum = torch.gather(e_affinity_sum, 1, li_index.long())
        # weights = e_affinity / (e_affinity_sum + 1e-3)
        # print("weighs:",weights)
        weights = self.wsoftmax(affinity)
        retrieved_output_flatten = torch.einsum('bn,bnc->bnc', weights, v)
        retrieved_output_flatten = self.output(retrieved_output_flatten).permute(0, 2, 1)#(1,64,4096)
        # img_feature = unsorted_segment_sum_pytorch(retrieved_output_flatten, li_index)
        # print("img_feature:", retrieved_output_flatten.size())  # B,C,N(1,64,4096)
        output_add_norm=self.bni(retrieved_output_flatten+img_feature_fusion.permute(0,2,1))
        # print("output_add_norm",output_add_norm.size())
        FC1 = self.fc1(output_add_norm.permute(0,2,1))
        relu = self.relu(FC1)
        FC2=self.fc2(relu).permute(0,2,1)
        output=self.bni(output_add_norm+FC2)
        # output = self.bni(output_add_norm+self.fc2(self.relu(self.fc1(output_add_norm))))
        return output


class Crossalign_k(nn.Module):
    def __init__(self, image_c, Lidar_c, qkv_c=128):
        super().__init__()
        self.ic, self.pc, self.qkv = image_c, Lidar_c, qkv_c
        self.q_embedding = nn.Linear(self.pc, self.qkv)  # lidar->q
        self.k_embedding = nn.Linear(self.ic, self.qkv)  # k
        self.v_embedding = nn.Linear(self.ic, self.qkv)  # v
        self.wsoftmax = nn.Softmax(dim=-1)
        # self.i2p = nn.Linear(self.ic,self.pc)
        self.output = nn.Linear(self.qkv, self.ic)
        # self.dropout= nn.Dropout(0.7)

    def forward(self, img_feas, point_feas, li_index, li_xyz):
        # img_feas(B,C,H,W)
        # point_feas(B,C,N)
        # li_index(B,N)
        # li_xyz(B,N,3)
        # print("li_index:",li_index.size())
        knn_neibor = k_nearest_neighbor(li_xyz, li_xyz, 4)  # (B,N,K)
        img_feas = img_feas.permute(0, 2, 3, 1)  # B,H,W,C
        flat_img = img_feas.reshape(img_feas.size(0), -1, img_feas.size(3))
        li_index_expanded = li_index.unsqueeze(-1).expand(-1, -1, self.ic).long()
        # print("li_index:",li_index_expanded.size())
        # print("flat_img:", flat_img.size())
        img_feature_fusion = torch.gather(flat_img, 1, li_index_expanded)  # B N C
        point_feas = point_feas.transpose(1, 2).contiguous()  # B,N,C
        img_feature_fusion = img_feature_fusion.unsqueeze(2).expand(knn_neibor.size(0), knn_neibor.size(1),
                                                                    knn_neibor.size(2),
                                                                    img_feature_fusion.size(2))  # B,N,K,C
        # print("img_feature_fusion_expand:", img_feature_fusion.size())  # B N  K C
        # print("knn_neibor:",knn_neibor.size())
        knn_neibor_d = knn_neibor.unsqueeze(-1).expand(knn_neibor.size(0), knn_neibor.size(1), knn_neibor.size(2),
                                                       img_feature_fusion.size(3))
        # print("knn_neibor_d",knn_neibor_d.size())
        img_feature_fusion = torch.gather(img_feature_fusion, 1, knn_neibor_d)
        # print("img_feature_fusion_index:", img_feature_fusion.size())  # B N  K C
        point_feature_expand = point_feas.unsqueeze(2).expand(point_feas.size(0), point_feas.size(1),
                                                              knn_neibor.size(2), point_feas.size(2))  # B,N,K,C
        # print("point_feature:",point_feature_expand.size())
        knn_neibor_l = knn_neibor.unsqueeze(-1).expand(knn_neibor.size(0), knn_neibor.size(1), knn_neibor.size(2),
                                                       point_feature_expand.size(3))
        point_feature_index = torch.gather(point_feature_expand, 1, knn_neibor_l)
        # print("point_feature_index:", point_feature_index.size())  # B N  K C
        # print("point_feas:",point_feas.size()) #[2,96,4096] B C N
        # print("img_feature_fusion:",img_feature_fusion.size())#B N C
        # img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        # point_feas_f = point_feas.transpose(1, 2).contiguous()  # BCN->BNC->(BN)C'
        q = self.q_embedding(point_feature_index)
        k = self.k_embedding(img_feature_fusion)
        v = self.v_embedding(img_feature_fusion)
        # print("q:",q.size()) 2,4096,128
        # print("k:", k.size())B,N,C
        # print("v:", v.size())B,N,C
        affinity = torch.einsum('bnkc,bnkc->bnk', q, k) / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
        # print("affinity:", affinity.size())  # B,N
        # maxaffinity = unsorted_segment_max_pytorch(affinity, li_index)
        # # print(maxaffinity.size())
        # # print(li_index.size())
        # maxaffinity = torch.gather(maxaffinity, 1, li_index.long())
        # e_affinity = torch.exp(affinity - maxaffinity)
        # # print(e_affinity.size())
        # e_affinity_sum = unsorted_segment_sum_pytorch(e_affinity, li_index)
        # e_affinity_sum = torch.gather(e_affinity_sum, 1, li_index.long())
        # weights = e_affinity / (e_affinity_sum + 1e-3)
        # print("weighs:",weights)
        weights = self.wsoftmax(affinity)
        retrieved_output_flatten = torch.einsum('bnk,bnkc->bnkc', weights, v)
        retrieved_output_flatten = self.output(retrieved_output_flatten)
        # print("output:",retrieved_output_flatten.size()) # B,N,K,C

        # squared_sum = torch.sum(retrieved_output_flatten ** 2,dim=2)
        # norms = torch.sqrt(squared_sum)
        # # norms = torch.linalg.norm(retrieved_output_flatten,dim=3)
        # max_norm_indices= torch.argmax(norms,dim=2)
        # max_norm_indices=max_norm_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,retrieved_output_flatten.size(3))
        # # print("max_norm_indices",max_norm_indices.size())
        # retrieved_output_flatten = torch.gather(retrieved_output_flatten,2,max_norm_indices).squeeze(2)
        retrieved_output_flatten = torch.mean(retrieved_output_flatten, dim=2)
        # print("output:",retrieved_output_flatten.size()) # B,N,K,C->BNC
        # img_feature = unsorted_segment_sum_pytorch(retrieved_output_flatten, li_index)
        # print("img_feature:", retrieved_output_flatten.size())  # B,N,C
        output = retrieved_output_flatten.permute(0, 2, 1)
        # print("output:", output.size())  # B,N,K,C->BNC
        return output


class CrossAttentionFusionP2I(nn.Module):
    def __init__(self, img_dim, point_dim, num_heads):
        super(CrossAttentionFusionP2I, self).__init__()
        self.convp2i = nn.Conv2d(point_dim, img_dim, kernel_size=1)
        self.cross_attention = nn.MultiheadAttention(embed_dim=img_dim, num_heads=num_heads)
        self.conv = nn.Conv2d(img_dim * 2, img_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(img_dim)
        self.relu = nn.ReLU()

    def forward(self, project_point2img_feature, image):
        # 将特征图展平以适应多头注意力机制的输入格式 (B, C, H, W) -> (B, H*W, C)

        project_point2img_feature = self.relu(self.bn(self.convp2i(project_point2img_feature)))
        gather_feature = project_point2img_feature+image
        B, C, H, W = project_point2img_feature.size()
        _, Ci, _, _ = image.size()
        project_point2img_feature_flat = gather_feature.view(B, C, -1).permute(0, 2, 1)
        image_flat = image.view(B, Ci, -1).permute(0, 2, 1)
        # print("pro", project_point2img_feature.size())
        # print("image", image.size())
        # print("pro_flat",project_point2img_feature_flat.size())
        # print("image_flat", image_flat.size())
        # 交叉注意力机制
        attn_output, _ = self.cross_attention(query=project_point2img_feature_flat, key=image_flat, value=image_flat)

        # 将结果重新 reshape 回原来的尺寸
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)

        # 拼接原始图像特征和注意力输出
        fusion_features = torch.cat([attn_output, image], dim=1)

        # 卷积、归一化和激活
        fusion_features = self.conv(fusion_features)

        return fusion_features


class Fusion_Cross_Conv_Gate(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        print('##############USE Fusion_Cross_Conv_Gate(ADD)#########')
        super(Fusion_Cross_Conv_Gate, self).__init__()
        self.P2IA_Layer = P2IA_Layer(channels=[inplanes_I, inplanes_P])
        self.inplanes = inplanes_I + inplanes_P
        self.outplanes = outplanes
        self.conv1 = conv3x3(self.inplanes, self.outplanes, stride=1)
        self.bn1 = BatchNorm2d(self.outplanes)
        # self.fusion = CrossAttentionFusionP2I(inplanes_I,inplanes_P,8)

    def forward(self, point_features, img_features, li_xy_cor, image,li_xyz):
        point_features = self.P2IA_Layer(img_features, point_features,li_xyz)
        # print("point_features", point_features.size())
        project_point2img_feature = grid_sample_reverse(point_features, li_xy_cor, img_shape=image.shape)
        # print("project", project_point2img_feature.size())
        # print("img", image.size())
        fusion_features = torch.cat([project_point2img_feature, image], dim=1)
        #
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        #fusion_features = self.fusion(project_point2img_feature, image)
        return fusion_features


# class IA_Layer(nn.Module):
#     def __init__(self, channels):
#         super(IA_Layer, self).__init__()
#         self.ic, self.pc = channels
#         rc = self.pc // 4
#         self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.rc, 1),  #####
#                                    nn.BatchNorm1d(rc),  ####
#                                    nn.ReLU())
#         self.fcK = nn.Linear(self.ic, rc)  # K
#         self.fcV = nn.Linear(self.ic, rc)  # V
#         self.fcQ = nn.Linear(self.pc, rc)  # Q
#         self.fc3 = nn.Linear(rc, 1)
#         self.fcout = nn.Linear(rc, self.pc)
#
#         ############
#         # self.q_embedding = nn.Linear(self.ic, rc)
#         # self.k_embedding = nn.Linear(self.pc, rc)
#         # self.v_embedding = nn.Linear(self.pc, rc)
#         # self.output_embedding = nn.Linear(rc, self.pc)
#         # self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, img_feas, point_feas):
#         batch = img_feas.size(0)
#         img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
#         point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
#         # print("img_feas_f:",img_feas_f.size())
#         # print("point_feas_f:",point_feas_f.size())
#         # query = self.q_embedding(img_feas_f)
#         # key = self.k_embedding(point_feas_f)
#         # value = self.v_embedding(point_feas_f)
#         # # 计算点云特征和图像特征之间的注意力权重
#         # attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
#         #     torch.tensor(query.size(-1), dtype=torch.float32))
#         # attention_weights = self.softmax(attention_scores)  # (B, N, N)
#         #
#         # # 使用注意力权重加权图像特征
#         # weighted_img_features = torch.matmul(attention_weights, value)  # (B, N, D)
#         #
#         # # 输出融合后的图像特征
#         # out= self.output_embedding(weighted_img_features)  # (B, N, C_i)
#         # out = out.transpose(1,2)
#         # print("out:",out.size())
#         # print(img_feas)
#         #####################
#         # ri = self.fc1(img_feas_f)
#         # rp = self.fc2(point_feas_f)
#         # att = F.sigmoid(self.fc3(F.tanh(ri + rp)))  # BNx1
#         # print("att:",att.size())
#         # att = att.squeeze(1)
#         # att = att.view(batch, 1, -1)  # B1N
#         # print("attview:", att.size())
#         # # print(img_feas.size(), att.size())
#         #
#         # print("img_feas:", img_feas.size())
#         # print("point_feas:", point_feas.size())
#         # img_feas_new = self.conv1(img_feas)
#         # print("img_feas_new:", img_feas_new.size())
#         # out = img_feas_new * att
#         # print("out",out.size())
#         ################
#         K = self.fcK(img_feas_f)
#         V = self.fcV(img_feas_f)
#         Q = self.fcQ(point_feas_f)
#         rc = self.pc // 4
#         att_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / K.size(-1) ** 0.5, dim=-1)
#         att_output = torch.matmul(att_scores, V)  # pc,rc
#         img_feas = self.conv1(img_feas)
#         img_feas_ff = img_feas.transpose(1, 2).contiguous().view(-1, rc)
#         fc_out = self.fcout(img_feas_ff + att_output)
#         out = fc_out.view(batch, -1, self.pc)
#         out = out.permute(0, 2, 1)
#         return out
class IA_Layer(nn.Module):
    def __init__(self, channels):
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),  #####
                                    nn.BatchNorm1d(self.pc),  ####
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc4 = nn.Linear(3,rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas,xyz):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        xyz_f=xyz.transpose(1,2).contiguous().view(-1, 3)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        rz = self.fc4(xyz_f)

        att = F.sigmoid(self.fc3(F.tanh(ri*rz + rp))) # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) # B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out

class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels=[inplanes_I, inplanes_P])
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        # self.conv2 = nn.Conv1d(outplanes,outplanes,3,padding=1,groups=outplanes)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)
        # self.bn2 = torch.nn.BatchNorm1d(outplanes)
        # self.relu = nn.ReLU()
        # self.res_conv = nn.Conv1d(outplanes,outplanes,1)

    def forward(self, point_features, img_features,xyz):
        img_features = self.IA_Layer(img_features, point_features,xyz)

        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        # print("fusion_features:",fusion_features.size())
        # fusion_features = self.relu(self.bn2(self.conv2(fusion_features)))
        # fusion_features = fusion_features+self.res_conv(fusion_features)

        return fusion_features


class SelfAttentionModule(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, query, key, value):
        # transpose to match nn.MultiheadAttention input shape requirements: (N, B, E)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # Apply multihead attention
        attn_output, attn_output_weights = self.attention(query, key, value)

        # Transpose back to (B, N, E)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def grid_sample_reverse(point_feature, xy, img_shape):
    # print('#######point_feature:', point_feature.shape)
    # print('#######xy:', xy.shape)
    # print('#######size:', size)
    size = [i for i in img_shape]
    size[1] = point_feature.shape[1]
    project_point2img = sample2BilinearGrid_F(point_feature, xy, size)

    return project_point2img


def get_model(input_channels=6, use_xyz=True):
    return Pointnet2MSG(input_channels=input_channels, use_xyz=use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        global layer
        super().__init__()
        # fidx_u = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
        # fidx_v = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]
        # PIC_C = [[64, 192, 640], [128, 96, 320], [256, 48, 160], [512, 24, 80]]
        # self.fft_modules = nn.ModuleList()
        # for (channels, width, height) in PIC_C:
        #     self.fft_modules.append(FACMA(channels, width, height, fidx_u, fidx_v))
        # CLF_LIST = [[64, 96], [128, 256], [256, 512], [512, 1024]]
        # self.clfusion = nn.ModuleList()
        # for (input2d, input3d) in CLF_LIST:
        #     self.clfusion.append(CLFM(input2d, input3d))
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        PIC_LIST = [[64, 96], [128, 256], [256, 512], [512, 1024]]
        self.Cross_align = nn.ModuleList()
        # # self.I2Ifusion = nn.ModuleList()
        for ic, pc in PIC_LIST:
            self.Cross_align.append(Crossalign(ic, pc))
            # self.I2Ifusion.append(FusionAwareInterpii(ic, pc))  # B,C,H,W

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            # if cfg.USE_SELF_ATTENTION:
            #     channel_out += cfg.RPN.SA_CONFIG.ATTN[k]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        if cfg.LI_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            self.Flatten = nn.ModuleList()  # 展平操作
            self.Multihaed = nn.ModuleList()  # 多头注意力机制
            self.Unflat = nn.ModuleList()  # 恢复形状
            if cfg.CROSS_FUSION:
                self.Cross_Fusion = nn.ModuleList()
            if cfg.USE_IM_DEPTH:
                cfg.LI_FUSION.IMG_CHANNELS[0] = cfg.LI_FUSION.IMG_CHANNELS[0] + 1

            if cfg.INPUT_CROSS_FUSION:
                cfg.LI_FUSION.IMG_CHANNELS[0] = cfg.LI_FUSION.IMG_CHANNELS[0] + 4

            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):

                self.Img_Block.append(
                    BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i + 1], stride=1)
                )

                if cfg.LI_FUSION.ADD_Image_Attention:
                    self.Fusion_Conv.append(
                        Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                          cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    self.Fusion_Conv.append(
                        Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                    cfg.LI_FUSION.POINT_CHANNELS[i]))

                if cfg.CROSS_FUSION:
                    if cfg.USE_P2I_GATE:
                        self.Cross_Fusion.append(
                            Fusion_Cross_Conv_Gate(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                                   cfg.LI_FUSION.IMG_CHANNELS[i + 1]))
                    else:
                        self.Cross_Fusion.append(
                            Fusion_Cross_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                              cfg.LI_FUSION.IMG_CHANNELS[i + 1]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                                                      kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                      stride=cfg.LI_FUSION.DeConv_Kernels[i]))
                # output_height = 384
                # output_width = 1280
                # embed_dim = cfg.LI_FUSION.DeConv_Reduce[i]
                # num_heads = 8
                # self.Flatten.append(Flatten())  # 展平操作
                # self.Multihaed.append(MultiheadAttentionBlock(embed_dim, num_heads))  # 多头注意力机制
                # self.Unflat.append(Unflatten(embed_dim, output_height, output_width))  # 恢复形状

            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce),
                                               cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4, kernel_size=1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4)

            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
                                                                cfg.LI_FUSION.IMG_FEATURES_CHANNEL,
                                                                cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(
                    cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
                    cfg.LI_FUSION.IMG_FEATURES_CHANNEL)

        if cfg.USE_SELF_ATTENTION:  ## set as False
            # ref: https://github.com/AutoVision-cloud/SA-Det3D/blob/main/src/models/backbones_3d/pointnet2_backbone.py
            # point-fsa from cfe
            print('##################USE_SELF_ATTENTION!!!!!!!! ')
            self.context_conv3 = PointContext3D(cfg.RPN.SA_CONFIG,
                                                IN_DIM=cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + cfg.RPN.SA_CONFIG.MLPS[2][1][
                                                    -1])
            self.context_conv4 = PointContext3D(cfg.RPN.SA_CONFIG,
                                                IN_DIM=cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + cfg.RPN.SA_CONFIG.MLPS[3][1][
                                                    -1])
            self.context_fusion_3 = Fusion_Conv(
                cfg.RPN.SA_CONFIG.ATTN[2] + cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + cfg.RPN.SA_CONFIG.MLPS[2][1][-1],
                cfg.RPN.SA_CONFIG.MLPS[2][0][-1] + cfg.RPN.SA_CONFIG.MLPS[2][1][-1])
            self.context_fusion_4 = Fusion_Conv(
                cfg.RPN.SA_CONFIG.ATTN[3] + cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + cfg.RPN.SA_CONFIG.MLPS[3][1][-1],
                cfg.RPN.SA_CONFIG.MLPS[3][0][-1] + cfg.RPN.SA_CONFIG.MLPS[3][1][-1])

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )
        # self.Cross_Fusion_Final = Fusion_Cross_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4 + cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        batch_size = xyz.shape[0]

        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]

            x = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            y = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0
            xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
            l_xy_cor = [xy]
            img = [image]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])
            # print("i",i)
            # print("li_features", li_features.size())
            # print("li_index",li_index.size())
            if cfg.LI_FUSION.ENABLED:
                li_index_1 = li_index.long().unsqueeze(-1).repeat(1, 1, 2)
                li_xy_cor = torch.gather(l_xy_cor[i], 1, li_index_1)

                image = self.Img_Block[i](img[i])
                # print("image",image.size())
                if cfg.CROSS_FUSION:
                    if cfg.USE_P2I_GATE:
                        # print("P2Igateimage:",image.size())

                        #first_img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))
                        first_img_gather_feature = self.Cross_align[i](image, li_features, li_index)  # B,C.N
                        # print(" first_img_gather_feature:",  first_img_gather_feature.size())
                        image = self.Cross_Fusion[i](li_features, first_img_gather_feature, li_xy_cor, image,li_xyz)
                    else:
                        img_shape = image.shape
                        project_point2img_feature = grid_sample_reverse(li_features, li_xy_cor, img_shape)
                        image = self.Cross_Fusion[i](project_point2img_feature, image)
                # li_xy_cor_permuted = li_xy_cor.permute(0, 2, 1)
                # f_image = self.I2Ifusion[i](li_xy_cor_permuted, image, li_features)  # fake 3d
                # fl_image, f_rgb = self.fft_modules[i](image, f_image)
                # print(image.shape)
                # img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))
                # print("img_gather_feature",img_gather_feature.size())
                img_gather_feature = self.Cross_align[i](image, li_features, li_index)  # B,C.N
                # print(img_gather_feature.size())

                li_features = self.Fusion_Conv[i](li_features, img_gather_feature,li_xyz)
                # li_xy_cor_permuted = li_xy_cor.permute(0,2,1)
                # image, li_features = self.clfusion[i](li_xy_cor_permuted, image, li_features)
                if cfg.USE_SELF_ATTENTION:
                    if i == 2:
                        # Get context visa self-attention
                        l_context_3 = self.context_conv3(batch_size, li_features, li_xyz)
                        # Concatenate
                        # li_features = torch.cat([li_features, l_context_3], dim=1)
                        li_features = self.context_fusion_3(li_features, l_context_3)
                    if i == 3:
                        # Get context via self-attention
                        l_context_4 = self.context_conv4(batch_size, li_features, li_xyz)
                        # Concatenate
                        # li_features = torch.cat([li_features, l_context_4], dim=1)
                        li_features = self.context_fusion_4(li_features, l_context_4)

                l_xy_cor.append(li_xy_cor)
                img.append(image)

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.LI_FUSION.ENABLED:
            DeConv = []
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                DeConv.append(self.DeConv[i](img[i + 1]))

            de_concat = torch.cat(DeConv, dim=1)
            # de_concat,l_features[0] = self.clfusionend[0](l_xy_cor[0].permute(0,2,1),de_concat,l_features[0])
            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
            # print("img_fusion:",img_fusion.size())B,C,W,H
            # print("xy_fusion:", xy.size())
            # print("l_feature",l_features[0].size())B,C,N
            index = xy[:, :, 0]
            # print("index:",index.size())B.N
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            # img_fusion_gather_feature = self.Cross_alignend[0](img_fusion, l_features[0], index)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature,l_xyz[0])

        if cfg.LI_FUSION.ENABLED:
            return l_xyz[0], l_features[0], img_fusion, l_xy_cor[0]
        else:
            return l_xyz[0], l_features[0], None, None


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
