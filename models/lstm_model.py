import torch
import torch.nn as nn
import random


class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.F = nn.Sequential()
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        combined = torch.cat([x, hidden], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        hidden_tilde = hidden + self.F(hidden)  # prediction
        next_hidden = hidden_tilde + K * (x - hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])

        return self.H, self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H


class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            # print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3), stride=stride, padding=1,
                               output_padding=output_padding),
            nn.GroupNorm(16, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride=2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride=1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2 * nf, stride=2)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):
    def __init__(self, nc=1, nf=32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2 * nf, nf, stride=2)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride=1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=(3, 3), stride=2, padding=1,
                                       output_padding=1)  # (nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride=1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride=1)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):
    def __init__(self, nc=64, nf=64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride=1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride=1)  # (32) x 32 x 32

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class Encoder(torch.nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.encoder_E = encoder_E().to(device)  # general encoder 64x64x1 -> 32x32x32
        # self.encoder_Ep = encoder_specific()  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_r = encoder_specific().to(device)

    def forward(self, input):
        input = self.encoder_E(input)  # general encoder 64x64x1 -> 32x32x32
        input_conv = self.encoder_r(input)
        return input_conv


class Decoder(torch.nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.decoder_r = decoder_specific().to(device)
        self.decoder_D = decoder_D().to(device)  # general decoder 32x32x32 -> 64x64x1

    def forward(self, input):
        decoded_Dr = self.decoder_r(input)
        out_image = torch.sigmoid(self.decoder_D(decoded_Dr))

        return out_image


if __name__ == '__main__':
    interval = 3
    teacher_forcing_ratio = 0.7
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    convcell_forward = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3, kernel_size=(3, 3),
                        device=device)
    convcell_reverse = ConvLSTM(input_shape=(16, 16), input_dim=64, hidden_dims=[128, 128, 64], n_layers=3, kernel_size=(3, 3),
                        device=device)
    encoder = Encoder(device)
    decoder = Decoder(device)
    input_tensor = torch.rand(16, 6, 1, 64, 64)
    target_tensor = torch.rand(16, 1, 1, 64, 64)
    len_in = input_tensor.shape[1]
    len_ta = target_tensor.shape[1]
    criterion = nn.MSELoss()
    loss = []

    # 0 1 2 3 4, 0 1 3 4为input, 2为target
    # forward: t-1(in_gt)->t(in_pred), 其中t<interval
    for ii in range(interval - 1):
        print('forward_input')
        encoder_output1 = encoder(input_tensor[:, ii])
        hidden1, output1 = convcell_forward(encoder_output1, ii == 0)
        decoder_output1 = decoder(output1[-1])
        loss_forward = criterion(decoder_output1, input_tensor[:, ii + 1])
        print(loss_forward)
        loss.append(loss_forward)

    # reverse: t+1(in_gt)->t(in_pred), 其中t>interval
    for ij in range(interval - 1):
        print('reverse_input')
        encoder_output2 = encoder(input_tensor[:, interval * 2 - 1 - ij])
        hidden2, output2 = convcell_forward(encoder_output2, ij == 0)
        decoder_output1 = decoder(output2[-1])
        loss_reverse = criterion(decoder_output1, input_tensor[:, interval * 2 - 2 - ij])
        print(loss_reverse)
        loss.append(loss_reverse)

    # forward pred
    lstm_for_input = input_tensor[:, interval - 1]      # 正向预测的第一帧由预测的前一帧输入得到
    for ti in range(target_tensor.shape[1]):
        print('forward prediction')
        encoder_forward_pred = encoder(lstm_for_input)
        hidden_forward_pred, output_forward_pred = convcell_forward(encoder_forward_pred, True)
        decoder_forward_pred = decoder(output_forward_pred[-1])
        loss_forward_pred = criterion(decoder_forward_pred, target_tensor[:, ti])
        print(loss_forward_pred)
        loss.append(loss_forward_pred)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            lstm_for_input = target_tensor[:, ti]
        else:
            lstm_for_input = decoder_forward_pred

    # reverse: 对target逐帧从后往前进行预测
    lstm_rev_input = input_tensor[:, interval]      # 反向预测的第一帧由预测的后一帧输入得到
    for tj in range(target_tensor.shape[1]):
        print('reverse prediction')
        encoder_reverse_pred = encoder(lstm_rev_input)
        hidden_reverse_pred, output_reverse_pred = convcell_reverse(encoder_reverse_pred, True)
        decoder_reverse_pred = decoder(output_reverse_pred[-1])
        loss_reverse_pred = criterion(decoder_reverse_pred, target_tensor[:, tj])
        print(loss_reverse_pred)
        loss.append(loss_reverse_pred)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            lstm_input = target_tensor[:, tj]
        else:
            lstm_input = decoder_reverse_pred












