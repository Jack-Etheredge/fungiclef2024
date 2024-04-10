"""
https://arxiv.org/pdf/2104.02939v3.pdf

The MLP discriminator in OpenGANfea takes a
D-dimensional feature as the input. Its architecture has a set of fully-connected layers (fc marked
with input-dimension and output-dimension), Batch
Normalization layers (BN) and LeakyReLU layers (hyper-parameter as 0.2): fc (D→64*8),
BN, LeakyReLU, fc (64*8→64*4),
BN, LeakyReLU, fc (64*4→64*2),
BN, LeakyReLU, fc (64*2→64*1), BN,
LeakyReLU, fc (64*1→1), Sigmoid.

• The MLP generator synthesizes a D-dimensional
feature given a 64-dimensional random vector: fc (64→64*8), BN, LeakyReLU,
fc (64*8→64*4), BN, LeakyReLU,
fc (64*4→64*2), BN, LeakyReLU,
fc (64*2→64*4), BN, LeakyReLU, fc
(64*4→D), Tanh.
"""

from torch import nn


class LayerNormDiscriminator(nn.Module):

    def __init__(self, nc=512, hidden_dim=64, leaky_relu_neg_slope=0.2):
        super().__init__()
        self.nc = nc
        self.hidden_dim = hidden_dim
        self.lrns = leaky_relu_neg_slope

        self.model = nn.Sequential(
            nn.Linear(self.nc, self.hidden_dim * 8),
            nn.LayerNorm(self.hidden_dim * 8),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 4),
            nn.LayerNorm(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):

    def __init__(self, nc=512, hidden_dim=64, leaky_relu_neg_slope=0.2):
        super().__init__()
        self.nc = nc
        self.hidden_dim = hidden_dim
        self.lrns = leaky_relu_neg_slope

        self.model = nn.Sequential(
            nn.Linear(self.nc, self.hidden_dim * 8),
            nn.BatchNorm1d(self.hidden_dim * 8),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):

    def __init__(self, nz=100, hidden_dim=64, nc=512, leaky_relu_neg_slope=0.2):
        super().__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.lrns = leaky_relu_neg_slope

        self.model = nn.Sequential(
            nn.Linear(self.nz, self.hidden_dim * 8),
            nn.BatchNorm1d(self.hidden_dim * 8),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.nc),
            nn.ReLU(inplace=True),  # this is to match the output of the embedder model
        )

    def forward(self, input):
        return self.model(input)


class MLPDiscriminator(nn.Module):
    """
    The MLP discriminator in OpenGANfea takes a
    D-dimensional feature as the input. Its architecture has a set of fully-connected layers (fc marked
    with input-dimension and output-dimension), Batch
    Normalization layers (BN) and LeakyReLU layers (hyper-parameter as 0.2): fc (D→64*8),
    BN, LeakyReLU, fc (64*8→64*4),
    BN, LeakyReLU, fc (64*4→64*2),
    BN, LeakyReLU, fc (64*2→64*1), BN,
    LeakyReLU, fc (64*1→1), Sigmoid.
    """

    def __init__(self, nc=512, hidden_dim=64, leaky_relu_neg_slope=0.2):
        super().__init__()
        self.nc = nc
        self.hidden_dim = hidden_dim
        self.lrns = leaky_relu_neg_slope

        self.model = nn.Sequential(
            nn.Linear(self.nc, self.hidden_dim * 8),
            nn.BatchNorm1d(self.hidden_dim * 8),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


class MLPGenerator(nn.Module):
    """
    • The MLP generator synthesizes a D-dimensional
    feature given a 64-dimensional random vector: fc (64→64*8), BN, LeakyReLU,
    fc (64*8→64*4), BN, LeakyReLU,
    fc (64*4→64*2), BN, LeakyReLU,
    fc (64*2→64*4), BN, LeakyReLU, fc
    (64*4→D), Tanh.
    """

    def __init__(self, nz=100, hidden_dim=64, nc=512, leaky_relu_neg_slope=0.2):
        super().__init__()
        self.nz = nz
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.lrns = leaky_relu_neg_slope

        self.model = nn.Sequential(
            nn.Linear(self.nz, self.hidden_dim * 8),
            nn.BatchNorm1d(self.hidden_dim * 8),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.BatchNorm1d(self.hidden_dim * 4),
            nn.LeakyReLU(self.lrns, inplace=True),
            nn.Linear(self.hidden_dim * 4, self.nc),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class CNNGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=512):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.Conv2d(self.nz, self.ngf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d(self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d(self.ngf * 4, self.nc, 1, 1, 0, bias=True),
            # nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        return self.model(input)


class CNNDiscriminator(nn.Module):
    def __init__(self, nc=512, ndf=64):
        super().__init__()
        self.nc = nc
        self.ndf = ndf
        self.model = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf * 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 8, self.ndf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
