from torch.nn import functional as F
from torch import nn
import torch
from mobilevit_v3.mobilevit import get_mobilevit_v3_xxs, get_mobilevit_v3_xs, get_mobilevit_v3_s
import math

class IRB_(nn.Module):
    """
    act_layer=nn.Hardswish
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                              groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # hardwish 激活函数
        x = self.act(x)
        # 3*3采样
        x = self.conv(x)
        # 激活函数
        x = self.act(x)
        # 把通道还原
        x = self.fc2(x)
        # 变回1维
        return x


class GIRB(nn.Module):
    def __init__(self, dim, last_dim, num_head, attn_drop=0.0, bias=True, patch_h: int = 8, patch_w: int = 8,
                 irb_expand_ratio=4):
        super().__init__()
        self.dim = dim
        self.last_dim = last_dim
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.patch_area = self.patch_w * self.patch_h
        self.attn_unit = CrossAttention(dim, num_heads=num_head, drop_rate=attn_drop, bias=bias)
        self.dim_conv = nn.Conv2d(last_dim, dim, kernel_size=1, bias=bias)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = IRB_(dim, irb_expand_ratio * dim)
        self.norm3 = nn.BatchNorm2d(dim)

    def forward(self, x, y):
        B, C, H, W = x.shape
        y = self.dim_conv(y)
        y_up = F.interpolate(y, (H, W), mode='bilinear')
        x, info_dict = unfolding(x, self.patch_h, self.patch_w)
        x = self.norm1(x)
        y, info_dict_y = unfolding(y, self.patch_h, self.patch_w)
        y = self.norm2(y)
        out = self.attn_unit(x, y)
        out = folding(out, self.patch_h, self.patch_w, info_dict)
        out = y_up + out
        out = out + self.mlp(self.norm3(out))
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, drop_rate=0., bias: bool = True, ):
        super().__init__()
        self.q_proj = nn.Linear(in_features=dim, out_features=dim, bias=bias)
        self.kv_proj = nn.Linear(in_features=dim, out_features=2 * dim, bias=bias)
        self.attn_dropout = nn.Dropout(p=drop_rate)
        self.out_proj = nn.Linear(in_features=dim, out_features=dim, bias=bias)
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = dim

    def forward(self, x, y):
        B, N, C = x.shape
        B2, N2, C2 = y.shape
        q = self.q_proj(x)
        q = q.reshape(B, N, self.num_heads, -1).transpose(1, 2).contiguous()
        kv = self.kv_proj(y)
        kv = kv.reshape(B2, N2, 2, self.num_heads, -1).transpose(1, 3).contiguous()
        k, v = kv[:, :, 0], kv[:, :, 1]
        q = q * self.scaling
        k = k.transpose(-1, -2)
        attn = torch.matmul(q, k)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        # print(attn.shape)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)
        return out

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=True, group=1, dilation=1,
                 act=nn.PReLU()):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias, groups=group,
                      dilation=dilation),
            nn.BatchNorm2d(out_channel),
            act)

    def forward(self, x):
        # print(x.shape)
        return self.conv(x)

class IRB_LW1ConvV4(nn.Module):
    """
    act_layer=nn.Hardswish
    在IRB出口处也使用相邻求和再投射的方式+残差
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, in_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                              groups=hidden_features)
        self.fc2 = nn.Conv2d(out_features, out_features, 1, 1, 0)
        self.alpha = nn.Parameter(torch.ones((in_features))*0.5, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((in_features))*0.5, requires_grad=True)

    def forward(self, x):
        res = x
        x_ = self.fc1(x)
        x = torch.cat([x, x_], dim=1)
        # hardwish 激活函数
        x = self.act(x)
        # 3*3采样
        x = self.conv(x)
        # 激活函数
        x = self.act(x)
        # print(x.shape)
        x = (self.alpha * x[:, 0:-1:2, :, :].permute(0, 2, 3, 1) + self.gamma * x[:, 1::2, :, :].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # print(self.alpha, self.gamma)
        # print(x.shape)
        # 把通道还原
        x = self.fc2(x)
        # 变回1维
        return x+res

class SMC(nn.Module):
    def __init__(self, in_channel, act=nn.PReLU()):
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=2, dilation=2)
        self.conv7x7 = nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=3, dilation=3)
        self.bn = nn.BatchNorm2d(in_channel)
        self.act = act

    def forward(self, x):
        B, C, H, W = x.shape
        x_1 = self.conv1x1_1(x[:, :int(C * 0.25), :, :])
        x_2 = self.conv3x3(x[:, int(C * 0.25):C // 2, :, :])
        x_3 = self.conv5x5(x[:, C // 2:int(C * 0.75), :, :])
        x_4 = self.conv7x7(x[:, int(C * 0.75):, :, :])
        x = self.conv1x1_2(torch.cat([x_1, x_2, x_3, x_4], dim=1))
        x = self.bn(x)
        x = self.act(x)
        return x


def unfolding(x, patch_h, patch_w):
    patch_w, patch_h = patch_w, patch_h
    patch_area = patch_w * patch_h
    batch_size, in_channels, orig_h, orig_w = x.shape

    new_h = int(math.ceil(orig_h / patch_h) * patch_h)
    new_w = int(math.ceil(orig_w / patch_w) * patch_w)

    interpolate = False
    if new_w != orig_w or new_h != orig_h:
        # Note: Padding can be done, but then it needs to be handled in attention function.
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        interpolate = True

    # number of patches along width and height
    num_patch_w = new_w // patch_w  # n_w
    num_patch_h = new_h // patch_h  # n_h
    num_patches = num_patch_h * num_patch_w  # N

    # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
    x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
    # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
    x = x.transpose(1, 2)
    # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
    x = x.reshape(batch_size, in_channels, num_patches, patch_area)
    # [B, C, N, P] -> [B, P, N, C]
    x = x.transpose(1, 3)
    # [B, P, N, C] -> [BP, N, C]
    x = x.reshape(batch_size * patch_area, num_patches, -1)

    info_dict = {
        "orig_size": (orig_h, orig_w),
        "batch_size": batch_size,
        "interpolate": interpolate,
        "total_patches": num_patches,
        "num_patches_w": num_patch_w,
        "num_patches_h": num_patch_h,
    }

    return x, info_dict


def folding(x, patch_h, patch_w, info_dict):
    n_dim = x.dim()
    patch_area = patch_h * patch_w
    assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
        x.shape
    )
    # [BP, N, C] --> [B, P, N, C]
    x = x.contiguous().view(
        info_dict["batch_size"], patch_area, info_dict["total_patches"], -1
    )

    batch_size, pixels, num_patches, channels = x.size()
    num_patch_h = info_dict["num_patches_h"]
    num_patch_w = info_dict["num_patches_w"]

    # [B, P, N, C] -> [B, C, N, P]
    x = x.transpose(1, 3)
    # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
    x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, patch_h, patch_w)
    # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
    x = x.transpose(1, 2)
    # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
    x = x.reshape(batch_size, channels, num_patch_h * patch_h, num_patch_w * patch_w)
    if info_dict["interpolate"]:
        x = F.interpolate(
            x,
            size=info_dict["orig_size"],
            mode="bilinear",
            align_corners=False,
        )
    return x

class HASNetXS(nn.Module):
    """
        v43+IRB残差
    """

    def __init__(self):
        super().__init__()
        self.backbone = get_mobilevit_v3_xs(load=True)
        self.girb5_4 = GIRB(160, 160, 4, irb_expand_ratio=2)
        self.girb5_3 = GIRB(96, 160, 4, irb_expand_ratio=2)

        self.girb4_3 = GIRB(96, 160, 4, irb_expand_ratio=2)

        self.dim_conv4_2 = nn.Conv2d(160, 48, 1)
        self.irb4_2 = IRB_LW1ConvV4(48, 48 * 2, 48)
        self.dim_conv3_2 = nn.Conv2d(96, 48, 1)
        self.irb_f3_2 = IRB_LW1ConvV4(48, 48 * 2, 48)
        self.dim_conv3_1 = nn.Conv2d(96, 32, 1)
        self.irb_f3_1 = IRB_LW1ConvV4(32, 32 * 2, 32)
        self.dim_conv2_1 = nn.Conv2d(48, 32, 1)
        self.irb_f2_1 = IRB_LW1ConvV4(32, 32 * 2, 32)

        self.conv53_43 = SMC(96)
        self.conv52_42_32 = SMC(48)
        self.conv51_41_31_21 = SMC(32)

        self.dim_conv1 = nn.Conv2d(160, 96, 1)
        self.irb_f4 = IRB_LW1ConvV4(96, 96 * 2, 96)
        self.dim_conv2 = nn.Conv2d(96, 48, 1)
        self.irb_f3 = IRB_LW1ConvV4(48, 48 * 2, 48)
        self.dim_conv3 = nn.Conv2d(48, 32, 1)
        self.irb_f2 = IRB_LW1ConvV4(32, 32 * 2, 32)

        self.sideout_4 = CBR(160, 1, 3, 1, 1, act=nn.PReLU())
        self.sideout_3 = CBR(96, 1, 3, 1, 1, act=nn.PReLU())
        self.sideout_2 = CBR(48, 1, 3, 1, 1, act=nn.PReLU())
        self.sideout_1 = CBR(32, 1, 3, 1, 1, act=nn.PReLU())

    def forward(self, x):
        w, h = x.size()[2:]
        _, x1, x2, x3, x4, x5, _ = self.backbone(x)

        x5_4 = self.girb5_4(x4, x5) + x4
        side_out4 = self.sideout_4(x5_4)
        x5_3 = self.girb5_3(x3, x5) + x3

        x4_3 = self.girb4_3(x3, x4) + x3
        x4_2 = self.irb4_2(x2 + F.interpolate(self.dim_conv4_2(x4), size=(x2.size()[2:]), mode='bilinear', ))

        x3_2 = self.irb_f3_2(x2 + F.interpolate(self.dim_conv3_2(x3), size=(x2.size()[2:]), mode='bilinear', ))
        x3_1 = self.irb_f3_1(x1 + F.interpolate(self.dim_conv3_1(x3), size=(x1.size()[2:]), mode='bilinear', ))
        x2_1 = self.irb_f2_1(x1 + F.interpolate(self.dim_conv2_1(x2), size=(x1.size()[2:]), mode='bilinear', ))

        x4 = self.conv53_43(x5_3 + x4_3)
        x3 = self.conv52_42_32(x4_2 + x3_2)
        x2 = self.conv51_41_31_21(x3_1 + x2_1)

        x4 = self.irb_f4(x4 + F.interpolate(self.dim_conv1(x5_4), size=(x4.size()[2:]), mode='bilinear', ))
        side_out3 = self.sideout_3(x4)
        x3 = self.irb_f3(x3 + F.interpolate(self.dim_conv2(x4), size=(x3.size()[2:]), mode='bilinear', ))
        side_out2 = self.sideout_2(x3)
        x2 = self.irb_f2(x2 + F.interpolate(self.dim_conv3(x3), size=(x2.size()[2:]), mode='bilinear', ))
        side_out1 = self.sideout_1(x2)
        side_out1 = F.interpolate(side_out1, size=(w, h), mode='bilinear', align_corners=True)
        return side_out4, side_out3, side_out2, side_out1

class HASNetS(nn.Module):
    """
        v43+s
    """

    def __init__(self):
        super().__init__()
        self.backbone = get_mobilevit_v3_s(load=True)
        self.girb5_4 = GIRB(256, 320, 4, irb_expand_ratio=2)
        self.girb5_3 = GIRB(128, 320, 4, irb_expand_ratio=2)

        self.girb4_3 = GIRB(128, 256, 4, irb_expand_ratio=2)

        self.dim_conv4_2 = nn.Conv2d(256, 64, 1)
        self.irb4_2 = IRB_LW1ConvV4(64, 64 * 2, 64)
        self.dim_conv3_2 = nn.Conv2d(128, 64, 1)
        self.irb_f3_2 = IRB_LW1ConvV4(64, 64 * 2, 64)
        self.dim_conv3_1 = nn.Conv2d(128, 32, 1)
        self.irb_f3_1 = IRB_LW1ConvV4(32, 32 * 2, 32)
        self.dim_conv2_1 = nn.Conv2d(64, 32, 1)
        self.irb_f2_1 = IRB_LW1ConvV4(32, 32 * 2, 32)

        self.conv53_43 = SMC(128)
        self.conv52_42_32 = SMC(64)
        self.conv51_41_31_21 = SMC(32)

        self.dim_conv1 = nn.Conv2d(256, 128, 1)
        self.irb_f4 = IRB_LW1ConvV4(128, 128 * 2, 128)
        self.dim_conv2 = nn.Conv2d(128, 64, 1)
        self.irb_f3 = IRB_LW1ConvV4(64, 64 * 2, 64)
        self.dim_conv3 = nn.Conv2d(64, 32, 1)
        self.irb_f2 = IRB_LW1ConvV4(32, 32 * 2, 32)

        self.sideout_4 = CBR(256, 1, 3, 1, 1, act=nn.PReLU())
        self.sideout_3 = CBR(128, 1, 3, 1, 1, act=nn.PReLU())
        self.sideout_2 = CBR(64, 1, 3, 1, 1, act=nn.PReLU())
        self.sideout_1 = CBR(32, 1, 3, 1, 1, act=nn.PReLU())

    def forward(self, x):
        w, h = x.size()[2:]
        _, x1, x2, x3, x4, x5, _ = self.backbone(x)
        # print(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
        x5_4 = self.girb5_4(x4, x5) + x4
        side_out4 = self.sideout_4(x5_4)
        x5_3 = self.girb5_3(x3, x5) + x3

        x4_3 = self.girb4_3(x3, x4) + x3
        x4_2 = self.irb4_2(x2 + F.interpolate(self.dim_conv4_2(x4), size=(x2.size()[2:]), mode='bilinear', ))

        x3_2 = self.irb_f3_2(x2 + F.interpolate(self.dim_conv3_2(x3), size=(x2.size()[2:]), mode='bilinear', ))
        x3_1 = self.irb_f3_1(x1 + F.interpolate(self.dim_conv3_1(x3), size=(x1.size()[2:]), mode='bilinear', ))
        x2_1 = self.irb_f2_1(x1 + F.interpolate(self.dim_conv2_1(x2), size=(x1.size()[2:]), mode='bilinear', ))

        x4 = self.conv53_43(x5_3 + x4_3)
        x3 = self.conv52_42_32(x4_2 + x3_2)
        x2 = self.conv51_41_31_21(x3_1 + x2_1)

        x4 = self.irb_f4(x4 + F.interpolate(self.dim_conv1(x5_4), size=(x4.size()[2:]), mode='bilinear', ))
        side_out3 = self.sideout_3(x4)
        x3 = self.irb_f3(x3 + F.interpolate(self.dim_conv2(x4), size=(x3.size()[2:]), mode='bilinear', ))
        side_out2 = self.sideout_2(x3)
        x2 = self.irb_f2(x2 + F.interpolate(self.dim_conv3(x3), size=(x2.size()[2:]), mode='bilinear', ))
        side_out1 = self.sideout_1(x2)
        side_out1 = F.interpolate(side_out1, size=(w, h), mode='bilinear', align_corners=True)
        return side_out4, side_out3, side_out2, side_out1
