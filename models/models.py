import torch
import torch.nn as nn
import torch.nn.functional as F
from config_para import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels=1):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            DoubleConv(n_channels, 8),
            Down(8, 16),
            Down(16, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512)
        )

    def forward(self, x):
        N, B, C, H, W = x.shape
        res = torch.zeros([N, B, 512, int(H / 64), int(W / 64)]).to(device)
        for i in range(N):
            res[i] = self.encoder(x[i])
        # torch.Size([4, 128, 512, 1, 1])
        return res


class Decoder(nn.Module):
    def __init__(self, n_classes=1):
        super(Decoder, self).__init__()

        self.n_classes = n_classes
        self.decoder = nn.Sequential(
            Up(512, 256),
            Up(256, 128),
            Up(128, 64),
            Up(64, 32),
            Up(32, 16),
            Up(16, 8),
            OutConv(8, n_classes),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x_input = x.view(self.batch_size, self.hidden_rnn, 1, 1)
        y = self.decoder(x)

        return y



class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=False,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        # self.hidden_prev = torch.zeros(num_layer, batch_size, hidden_size)
        self.linear = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x, h):

        out, h = self.rnn(x, h)          # 采取rnn中many--->one的形式
        pred = out[-1]
        # print(pred.shape)
        # exit()

        # 采取rnn中many--->many的形式,然后再将多个特征进行融合
        # fc_input = out.transpose(1, 0).reshape(x.shape[1], -1)
        # pred = self.linear(fc_input)

        return pred, h


class ODEfunc(nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.nfe = 0

    def forward(self, t, h):
        self.nfe += 1
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)

        return dh


if __name__ == '__main__':
    y = torch.randn(128, 1024, 4, 4)
    net = Decoder(1)
    out = net(y)
    print(out.shape)


    # x = torch.randn(4, 128, 1, 64, 64)
    # net = Encoder(1)
    # out = net(x)
    # print(1)
