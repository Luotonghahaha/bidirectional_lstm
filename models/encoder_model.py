import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Encoder(torch.nn.Module):
    # 编码器，将input_size维度数据压缩为latent_size维度
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):   # x: bs,input_size
        B, C, H, w = x.shape
        x = x.view(B, -1)
        x = F.relu(self.linear1(x))  #-> bs,hidden_size
        x = self.linear2(x)     #-> bs,latent_size
        return x


class Decoder(torch.nn.Module):
    # 解码器，将latent_size维度的压缩数据转换为output_size维度的数据
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):   # x:bs,latent_size
        x = F.relu(self.linear1(x))     # ->bs,hidden_size
        x = torch.sigmoid(self.linear2(x))  # ->bs,output_size
        return x


class ImageEncoder(nn.Module):
    def __init__(self, in_channel, hidden, out_layer):
        # out_layer is the output dim of linear
        super(ImageEncoder, self).__init__()
        self.in_channel = in_channel
        self.hidden_layer = hidden
        self.out_linear = out_layer
        self.in_linear = self.hidden_layer * 4 * 1 * 1
        self.conv1 = nn.Conv2d(self.in_channel, self.hidden_layer, 3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_layer, self.hidden_layer * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_layer * 2, self.hidden_layer * 4, 3, padding=1)
        self.fc1 = nn.Linear(self.in_linear, self.out_linear)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, self.in_linear)
        x = F.relu(self.fc1(x))
        return x

# model = ImageEncoder(3, 64, 128)
# x = torch.randn(10, 3, 8, 8)
# y = model(x)
# print(y.shape)   # torch.Size([10, 128])

class SymmetricEncoder(nn.Module):
    def __init__(self, in_channel, hidden, out_layer):
        # out_layer is the output dim of linear
        super(SymmetricEncoder, self).__init__()
        self.in_channel = in_channel
        self.hidden_layer = hidden
        self.out_linear = out_layer
        self.in_linear = self.hidden_layer * 4 * 1 * 1
        self.conv1 = nn.Conv2d(self.in_channel, self.hidden_layer, 3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_layer, self.hidden_layer * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_layer * 2, self.hidden_layer * 4, 3, padding=1)
        self.fc1 = nn.Linear(self.in_linear, self.out_linear)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, self.in_linear)
        x = F.relu(self.fc1(x))
        return x


class ImageDecoder(nn.Module):
    def __init__(self, out_channel, hidden, in_layer):
        # in_layer is the input dim of linear
        super(ImageEncoder, self).__init__()
        self.out_channel = out_channel
        self.hidden_layer = hidden
        self.in_layer = in_layer
        self.out_linear = self.hidden_layer * 4 * 1 * 1
        self.encoder = ImageEncoder(3, 64, 128)
        self.symmetric_encoder = SymmetricEncoder(3, 64, 128)
        self.fc2 = nn.Linear(self.in_layer, self.out_linear)
        self.deconv1 = nn.ConvTranspose2d(self.hidden_layer * 4, self.hidden_layer * 2, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(self.hidden_layer * 2, self.hidden_layer, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.hidden_layer, self.out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.view(-1, self.hidden_layer * 4, 1, 1)
        return x


# model = ImageDecoder(3, 64, 128)
# x = torch.randn(10, 128)
# y = model(x)

