# This file contains modules common to various models
import torch.nn as nn
import torch
import torch.nn.functional as F
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
      
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):#
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        w1_1 = torch.tensor([[[1., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w1_2 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[1., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w1_3 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[1., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w3_1 = torch.tensor([[[0., 1., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w3_2 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 1., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w3_3 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 1., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w2_1 = torch.tensor([[[0., 0., 0.],[1., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w2_2 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[1., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w2_3 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[1., 0., 0.],[0., 0., 0.]]])
        w4_1 = torch.tensor([[[0., 0., 0.],[0., 1., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w4_2 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 1., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]])
        w4_3 = torch.tensor([[[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]], [[0., 0., 0.],[0., 1., 0.],[0., 0., 0.]]])
        w1_1 = w1_1.view(1, 3, 3, 3)
        w1_2 = w1_2.view(1, 3, 3, 3)
        w1_3 = w1_3.view(1, 3, 3, 3)
        w2_1 = w2_1.view(1, 3, 3, 3)
        w2_2 = w2_2.view(1, 3, 3, 3)
        w2_3 = w2_3.view(1, 3, 3, 3)
        w3_1 = w3_1.view(1, 3, 3, 3)
        w3_2 = w3_2.view(1, 3, 3, 3)
        w3_3 = w3_3.view(1, 3, 3, 3)
        w4_1 = w4_1.view(1, 3, 3, 3)
        w4_2 = w4_2.view(1, 3, 3, 3)
        w4_3 = w4_3.view(1, 3, 3, 3)    
        self.w_cat = torch.cat([w1_1, w1_2,w1_3, w2_1,w2_2,w2_3, w3_1,w3_2,w3_3, w4_1,w4_2,w4_3], 0) 
        self.p2d = (0, 2, 0, 2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)      
        # x = x.type(torch.cuda.FloatTensor)
        #x_gt = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        x_pad = F.pad(x, self.p2d, 'constant', 0)
        xx = F.conv2d(x_pad, self.w_cat.to(x.device),stride=2) 
        xx = self.conv(xx)
        #print(torch.sum(x_gt - xx))
        return xx



class Focus_11(nn.Module):#
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # x = x.type(torch.cuda.FloatTensor)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# class C3(nn.Module):
#     # Cross Convolution CSP
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(C3, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))