import torch
import torch.nn as nn

from spikingjelly.activation_based import surrogate, layer, functional, neuron
from utilities import smart_padding, Conv2dWithConstraint, SmartPermute
from spiking_neuron import AQIFNode


# %% EEGNet
class EEGNet(nn.Module):
    """
    References
    ----------
    Lawhern V J, Solon A J, Waytowich N R, et al.
    EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J].
    Journal of neural engineering, 2018, 15(5): 056013.

    Parameters
    ----------
    chunk_size (int): Number of data points included in each EEG chunk, i.e.,: math:`T` in the paper.
    num_electrodes (int): The number of electrodes, i.e.,: math:`C` in the paper.
    F1 (int): The filter number of block 1, i.e.,: math:`F_1` in the paper.
    F2 (int): The filter number of block 2, i.e.,: math:`F_2` in the paper.
    D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper.
    num_classes (int): The number of classes to predict, i.e.,: math:`N` in the paper.
    sampling_rate (int): The sampling rate.
    dropout (float): Probability of an element to be zeroed in the dropout layers.
    """
    def __init__(
            self,
            chunk_size: int = 1000,
            num_electrodes: int = 22,
            F1: int = 8,
            F2: int = 16,
            D: int = 2,
            num_classes: int = 2,
            sampling_rate: int = 256,
            dropout: float = 0.25
    ):
        super(EEGNet, self).__init__()

        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_classes = num_classes
        self.kernel_1 = sampling_rate // 2
        self.kernel_2 = sampling_rate // 8
        self.drop_out = dropout

        self.block1 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel_1)),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.F1,
                kernel_size=(1, self.kernel_1),
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.num_electrodes, 1),
                max_norm=1,
                stride=1,
                padding=0,
                groups=self.F1,
                bias=False
            ),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=self.drop_out)
        )

        self.block2 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel_2)),
            nn.Conv2d(
                in_channels=self.F1 * self.D,
                out_channels=self.F1 * self.D,
                kernel_size=(1, self.kernel_2),
                stride=1,
                padding=0,
                groups=self.F1 * self.D,
                bias=False
            ),
            nn.Conv2d(
                in_channels=self.F1 * self.D,
                out_channels=self.F2,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=self.drop_out),
            nn.Flatten()
        )

        self.linear = nn.Linear(
            in_features=self.feature_dim(),
            out_features=self.num_classes,
            bias=False
        )

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
        return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)

        return x


# %% EEGNeX
class EEGNeX(nn.Module):
    """
    References
    ----------
    Chen X, Teng X, Chen H, Pan Y, Geyer P.
    Toward reliable signals decoding for electroencephalogram: A benchmark study to EEGNeX.
    Biomedical Signal Processing and Control. 2024 Jan 1;87:105475.

    Parameters
    ----------
    chunk_size (int): Number of data points included in each EEG chunk, i.e.,: math:`T` in the paper.
    num_electrodes (int): The number of electrodes, i.e.,: math:`C` in the paper.
    F1 (int): The filter number of block 1, i.e.,: math:`F_1` in the paper.
    F2 (int): The filter number of block 2, i.e.,: math:`F_2` in the paper.
    D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper.
    num_classes (int): The number of classes to predict, i.e.,: math:`N` in the paper.
    sampling_rate (int): The sampling rate.
    dropout (float): Probability of an element to be zeroed in the dropout layers.
    """
    def __init__(
            self,
            chunk_size: int = 1000,
            num_electrodes: int = 22,
            F1: int = 8,
            F2: int = 32,
            D: int = 2,
            num_classes: int = 2,
            dropout: float = 0.25
    ):
        super(EEGNeX, self).__init__()

        self.chunk_size = chunk_size
        self.kernel_1 = 32
        self.kernel_2 = 16
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_electrodes = num_electrodes
        self.dropout = dropout
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel_1)),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.F1,
                kernel_size=(1, self.kernel_1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.F1)
        )

        self.block2 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel_1)),
            nn.Conv2d(
                in_channels=self.F1,
                out_channels=self.F2,
                kernel_size=(1, self.kernel_1),
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.F2),
            nn.ELU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2 * self.D,
                kernel_size=(self.num_electrodes, 1),
                groups=self.F2,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.F2 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.F2 * self.D,
                out_channels=self.F2 * self.D,
                kernel_size=(1, self.kernel_2),
                dilation=(1, 2),
                padding = self.calc_padding((1, self.kernel_2), (1, 2)),
                bias=False
            ),
            nn.BatchNorm2d(self.F2 * self.D)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.F2 * self.D,
                out_channels=self.F1,
                kernel_size=(1, self.kernel_2),
                dilation=(1, 4),
                padding = self.calc_padding((1, self.kernel_2), (1, 4)),
                bias=False
            ),
            nn.BatchNorm2d(self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(p=self.dropout),
            nn.Flatten()
        )

        self.linear = nn.Linear(
            in_features=self.feature_dim(),
            out_features=self.num_classes,
            bias=False
        )

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)
            mock_eeg = self.block4(mock_eeg)
            mock_eeg = self.block5(mock_eeg)
        return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.linear(x)

        return x

    @staticmethod
    def calc_padding(kernel_size: tuple[int, int], dilation: tuple[int, int]) -> tuple[int, int]:
        """
        Calculate padding size for 'same' convolution with dilation.
        """
        padding_height = ((kernel_size[0] - 1) * dilation[0]) // 2
        padding_width = ((kernel_size[1] - 1) * dilation[1]) // 2
        return padding_height, padding_width


# %% EEG-DSNet
class EEG_DSNet(nn.Module):
    def __init__(
            self,
            chunk_size: int = 1000,
            num_electrodes: int = 22,
            filters: int = 8,
            frequency: int = 256,
            D: int = 2,
            num_classes: int = 2,
            T: int = 5,
            neuron_type: str = 'AQIFNode',
            pooling_type: str = 'AM'
    ):
        super(EEG_DSNet, self).__init__()


        self.kernel = 32
        self.filters = filters
        self.neuron_type = neuron_type
        self.T = T
        self.num_electrodes = num_electrodes
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.pooling_type = pooling_type

        self.temporal_branch = self.conv_block(filters=self.filters, temporal_branch=True)
        self.spectral_branch = self.conv_block(filters=self.filters * 2, temporal_branch=False)

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.feature_dim(),
                out_features=self.num_classes,
                bias=False
            ),
            self.spiking_neuron()
        )

        functional.set_step_mode(self, step_mode='m')

    def conv_block(self, filters: int, temporal_branch: bool = True):
        layers = [
            nn.ZeroPad2d(smart_padding(self.kernel)),
            layer.Conv2d(
                in_channels=1,
                out_channels=filters,
                kernel_size=(1, self.kernel),
                bias=False
            ),
            layer.BatchNorm2d(filters),
            self.spiking_neuron(),

            layer.Conv2d(
                in_channels=filters,
                out_channels=filters * self.D,
                kernel_size=(self.num_electrodes, 1),
                groups=filters,
                bias=False
            ),
            layer.BatchNorm2d(filters * self.D),
            self.spiking_neuron(),

            self.pooling_selection(
                kernel_size1=(1, 4),
                kernel_size2=(1, 4),
                pooling_type=self.pooling_type,
                temporal_branch=temporal_branch,
                ceil=False
            ),

            nn.ZeroPad2d(smart_padding(self.kernel // 2)),
            layer.Conv2d(
                in_channels=filters * self.D,
                out_channels=filters * self.D,
                kernel_size=(1, self.kernel // 2),
                groups=filters * self.D,
                bias=False
            ),
            layer.Conv2d(
                in_channels=filters * self.D,
                out_channels=filters * self.D,
                kernel_size=1,
                bias=False
            ),
            layer.BatchNorm2d(filters * self.D),
            self.spiking_neuron(),

            self.pooling_selection(
                kernel_size1=(1, 8),
                kernel_size2=(1, 16),
                pooling_type=self.pooling_type,
                temporal_branch=temporal_branch,
                ceil=True
            )
        ]
        if not temporal_branch:
            layers.append(SmartPermute())
        layers += [
            layer.Conv2d(
                in_channels=self.filters * self.D,
                out_channels=self.filters * self.D,
                kernel_size=(1, 6),
                dilation=(1, 2),
                padding=self.calc_padding((1, 6), (1, 2)),
                groups=self.filters * self.D,
                bias=False
            ),
            layer.Conv2d(
                in_channels=self.filters * self.D,
                out_channels=self.filters * self.D * 2,
                kernel_size=1,
                bias=False
            ),
            layer.Conv2d(
                in_channels=self.filters * self.D * 2,
                out_channels=self.filters * self.D * 2,
                kernel_size=(1, 6),
                dilation=(1, 4),
                padding=self.calc_padding((1, 6), (1, 4)),
                groups=self.filters * self.D * 2,
                bias=False
            ),
            layer.Conv2d(
                in_channels=self.filters * self.D * 2,
                out_channels=self.filters * self.D,
                kernel_size=1,
                bias=False
            ),
            layer.BatchNorm2d(self.filters * self.D),
            self.spiking_neuron(),
            layer.Flatten()
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batches, filters, electrodes, length)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, batches, filters, electrodes, length)
        x_temporal = self.temporal_branch(x)
        x_spectral = self.spectral_branch(x)
        x_concat = torch.cat((x_temporal, x_spectral), dim=2)
        x = self.linear(x_concat)
        x = x.mean(0)
        return x

    def spiking_encoder(self):
        return self.conv_block[1:]

    def spiking_neuron(self):
        if self.neuron_type == 'AQIFNode':
            return AQIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'IFNode':
            return neuron.IFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'LIFNode':
            return neuron.LIFNode(surrogate_function=surrogate.ATan())
        elif self.neuron_type == 'IzhikevichNode':
            return neuron.IzhikevichNode(surrogate_function=surrogate.ATan())
        else:
            raise ValueError(f"Unsupported surrogate function: {self.surrogate_function}")

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg_temporal = self.temporal_branch(mock_eeg)
            mock_eeg_spectral = self.spectral_branch(mock_eeg)
            mock_eeg = torch.cat((mock_eeg_temporal, mock_eeg_spectral), dim=1)
        return mock_eeg.shape[1]

    @staticmethod
    def calc_padding(kernel_size: tuple[int, int], dilation: tuple[int, int]) -> tuple[int, int]:
        """
        Calculate padding size for 'same' convolution with dilation.
        """
        padding_height = ((kernel_size[0] - 1) * dilation[0]) // 2
        padding_width = ((kernel_size[1] - 1) * dilation[1]) // 2
        return padding_height, padding_width

    @staticmethod
    def pooling_selection(
            kernel_size1: tuple[int, int],
            kernel_size2: tuple[int, int],
            pooling_type: str = 'AM',
            temporal_branch: bool = True,
            ceil: bool = False
    ):
        if pooling_type == 'AM':
            return layer.AvgPool2d(kernel_size=kernel_size1, ceil_mode=ceil) if temporal_branch \
                else layer.MaxPool2d(kernel_size=kernel_size2, ceil_mode=ceil)
        elif pooling_type == 'MA':
            return layer.MaxPool2d(kernel_size=kernel_size1, ceil_mode=ceil) if temporal_branch \
                else layer.AvgPool2d(kernel_size=kernel_size2, ceil_mode=ceil)
        elif pooling_type == 'MM':
            return layer.MaxPool2d(kernel_size=kernel_size1, ceil_mode=ceil) if temporal_branch \
                else layer.MaxPool2d(kernel_size=kernel_size2, ceil_mode=ceil)
        elif pooling_type == 'AA':
            return layer.AvgPool2d(kernel_size=kernel_size1, ceil_mode=ceil) if temporal_branch \
                else layer.AvgPool2d(kernel_size=kernel_size2, ceil_mode=ceil)
        else:
            raise ValueError(f"Unsupported surrogate function: {pooling_type}")


# %% CSNN
class CSNN(nn.Module):
    def __init__(
            self,
            chunk_size: int = 1000,
            num_electrodes: int = 22,
            F1: int = 12,
            F2: int = 64,
            num_classes: int = 2,
            kernel: int = 5,
            pool_size: int = 2,
            T: int = 10
    ):
        super(CSNN, self).__init__()

        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.num_classes = num_classes
        self.kernel = kernel
        self.pool_size = pool_size
        self.T = T

        self.block1 = nn.Sequential(
            layer.Conv2d(
                in_channels=1,
                out_channels=self.F1,
                kernel_size=self.kernel
            ),
            layer.MaxPool2d(self.pool_size),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), v_threshold=0.5)
        )

        self.block2 = nn.Sequential(
            layer.Conv2d(
                in_channels=self.F1,
                out_channels=self.F2,
                kernel_size=self.kernel
            ),
            layer.MaxPool2d(self.pool_size),
            layer.Flatten(),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), v_threshold=0.5)
        )

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.feature_dim(),
                out_features=self.num_classes
            ),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), v_threshold=0.5)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(1)  # (batches, virtual_channels, channels, length)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, batches, virtual_channels, channels, length)
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)
        x = x.mean(0)
        return x

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
        return mock_eeg.shape[1]

    def spiking_encoder(self):
        return self.block1[1:]


# %% SCNet
class SCNet(nn.Module):
    def __init__(
            self,
            chunk_size: int = 1000,
            num_electrodes: int = 22,
            F1: int = 8,
            F2: int = 16,
            num_classes: int = 2,
            kernel: int = 64,
            T: int = 5
    ):
        super(SCNet, self).__init__()

        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.num_classes = num_classes
        self.kernel = kernel
        self.T = T

        self.block1 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel)),
            layer.Conv2d(
                in_channels=1,
                out_channels=self.F1,
                kernel_size=(1, self.kernel),
                bias=False
            ),
            layer.BatchNorm2d(self.F1),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        self.block2 = nn.Sequential(
            layer.Conv2d(
                in_channels=self.F1,
                out_channels=self.F2,
                kernel_size=(self.num_electrodes, 1),
                groups=self.F1,
                bias=False
            ),
            layer.BatchNorm2d(self.F2),
            layer.MaxPool2d(kernel_size=(1, 4)),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        self.block3 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel // 4)),
            layer.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, self.kernel // 4),
                groups=self.F2,
                bias=False
            ),
            layer.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, 1),
                bias=False
            ),
            layer.BatchNorm2d(self.F2),
            layer.MaxPool2d(kernel_size=(1, 8)),
            layer.Flatten(),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=self.feature_dim(),
                out_features=self.num_classes
            ),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(1)  # (batches, virtual_channels, channels, length)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, batches, virtual_channels, channels, length)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.linear(x)
        x = x.mean(0)
        return x

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
            mock_eeg = self.block3(mock_eeg)
        return mock_eeg.shape[1]

    def spiking_encoder(self):
        return self.block1[1:]


# %% LENet
class LENet(nn.Module):
    def __init__(
            self,
            chunk_size: int = 1000,
            num_electrodes: int = 22,
            F1: int = 8,
            F2: int = 16,
            F3: int = 24,
            num_classes: int = 2,
            kernel: int = 64,
            T: int = 5
    ):
        super(LENet, self).__init__()

        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.num_classes = num_classes
        self.kernel = kernel
        self.T = T

        self.block1 = self.conv_block(filters=self.F1, kernel_size=self.kernel // 4)
        self.block2 = self.conv_block(filters=self.F1, kernel_size=self.kernel // 2)
        self.block3 = self.conv_block(filters=self.F1, kernel_size=self.kernel)

        self.block4 = nn.Sequential(
            layer.Conv2d(
                in_channels=self.F3,
                out_channels=self.F3,
                kernel_size=(1, 1),
                bias=False
            ),
            layer.BatchNorm2d(self.F3),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(
                in_channels=self.F3,
                out_channels=self.F2,
                kernel_size=(self.num_electrodes, 1),
                groups=self.F1,
                bias=False
            ),
            layer.BatchNorm2d(self.F2),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.AvgPool2d(kernel_size=(1, 4)),
            layer.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, self.kernel // 4),
                groups=self.F2,
                bias=False
            ),
            layer.Conv2d(
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=1,
                bias=False
            ),
            layer.BatchNorm2d(self.F2),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Conv2d(
                in_channels=self.F2,
                out_channels=self.num_classes,
                kernel_size=1,
                bias=False
            ),
            layer.AdaptiveAvgPool2d(1),
            layer.Flatten(),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(1)  # (batches, virtual_channels, channels, length)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, batches, virtual_channels, channels, length)
        x_1 = self.block1(x)
        x_2 = self.block2(x)
        x_3 = self.block3(x)
        x = torch.cat((x_1, x_2, x_3), dim=2)
        x = self.block4(x)
        x = x.mean(0)
        return x

    def spiking_encoder(self):
        return self.conv_block[1:]

    @staticmethod
    def conv_block(filters: int, kernel_size: int):
        layers = [
            nn.ZeroPad2d(smart_padding(kernel_size)),
            layer.Conv2d(
                in_channels=1,
                out_channels=filters,
                kernel_size=(1, kernel_size),
                bias=False
            ),
            layer.BatchNorm2d(filters),
            neuron.IFNode(surrogate_function=surrogate.ATan())
        ]
        return nn.Sequential(*layers)

# %% build_model
def build_model(args, config):
    if args.model_type == 'ANN':
        if args.model == 'EEGNet':
            model = EEGNet(
                chunk_size=config["chunk_size"],
                num_electrodes=config["num_electrodes"],
                sampling_rate=config["sampling_rate"]
            )
        elif args.model == 'EEGNeX':
            model = EEGNeX(
                chunk_size=config["chunk_size"],
                num_electrodes=config["num_electrodes"]
            )
        else:
            raise ValueError(f"Unsupported ANN model: {args.model}")
    elif  args.model_type == 'SNN':
        if args.model == 'SCNet':
            model = SCNet(
                chunk_size=config["chunk_size"],
                num_electrodes=config["num_electrodes"]
            )
        elif args.model == 'EEG_DSNet':
            model = EEG_DSNet(
                T=args.T,
                neuron_type=args.neuron_type,
                pooling_type=args.pooling_type,
                chunk_size=config["chunk_size"],
                num_electrodes=config["num_electrodes"]
            )
        elif args.model == 'CSNN':
            model = CSNN(
                chunk_size=config["chunk_size"],
                num_electrodes=config["num_electrodes"]
            )
        elif args.model == 'LENet':
            model = LENet(
                chunk_size=config["chunk_size"],
                num_electrodes=config["num_electrodes"]
            )
        else:
            raise ValueError(f"Unsupported SNN model: {args.model}")
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    return model.to(args.device)


# %%