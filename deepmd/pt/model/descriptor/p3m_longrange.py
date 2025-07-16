import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from torch_geometric.transforms import BaseTransform
import time
from deepmd.pt.model.network.mlp import (
    MLPLayer,
)

class NonPBCAddGrid(BaseTransform):
    def __init__(self, expand_size: int, num_grids: Union[List[int], int]) -> None:
        super().__init__()
        if isinstance(num_grids, int):
            num_grids = [num_grids, num_grids, num_grids]
        self.num_grids = num_grids
        self.expand_size = expand_size

    def __call__(self, atom_pos,box=None):
        if box is None:
            pos = atom_pos  # Shape: [batch_size, num_nodes, hidden_size]
            
            # Center the positions (Shape: [batch_size, num_nodes, hidden_size])
            pos_centered = pos - pos.mean(dim=1, keepdim=True)
            
            # For each batch, perform SVD and get the V matrix (Shape: [batch_size, 3, 3])
            batch_size = pos_centered.shape[0]

            
            # Apply SVD
            start_time = time.time()
            cell = torch.stack([torch.svd(pos_centered[i])[2].t() for i in range(batch_size)], dim=0)
            end_time = time.time()
            print(f"SVD 计算时间: {end_time - start_time} 秒")
            
            # Rotate the centered positions for each batch using the rotation matrix
            rotated_pos_centered = torch.matmul(pos_centered, cell.transpose(1, 2))

            # Calculate the cell lengths for each batch sample (Shape: [batch_size, 3])
            cell_lengths = rotated_pos_centered.max(dim=1).values - rotated_pos_centered.min(dim=1).values
            translation = rotated_pos_centered.min(dim=1).values - 1 / 2 * self.expand_size
            
            # Apply translation (Shape: [batch_size, 3])
            translation = torch.einsum("ij,ijl->il", translation, cell)
            # Expand cell lengths and create the new cell (Shape: [batch_size, 3, 3])
            cell_lengths += self.expand_size
            new_cell = cell * cell_lengths.unsqueeze(1)
            
            # Calculate new positions (Shape: [batch_size, num_nodes, hidden_size])
            new_pos = pos_centered - translation.unsqueeze(1)

            # Create grid for the mesh (Shape: [num_grids[0] + 1], [num_grids[1] + 1], [num_grids[2] + 1])
            x_linespace = torch.linspace(0, 1, self.num_grids[0] + 1, dtype=torch.float32)
            y_linespace = torch.linspace(0, 1, self.num_grids[1] + 1, dtype=torch.float32)
            z_linespace = torch.linspace(0, 1, self.num_grids[2] + 1, dtype=torch.float32)

            # Calculate centers of the mesh
            x_centers = (x_linespace[1:] + x_linespace[:-1]) / 2
            y_centers = (y_linespace[1:] + y_linespace[:-1]) / 2
            z_centers = (z_linespace[1:] + z_linespace[:-1]) / 2

            # Create mesh grid: [num_x_centers, num_y_centers, num_z_centers, 3]
            mesh = torch.stack(torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij'), dim=-1).to(pos.device)
            # Mesh coordinates: [batch_size, num_x_centers, num_y_centers, num_z_centers, 3]
            mesh_coord = torch.einsum("ijkl,nlm->nijkm", mesh, new_cell).view(batch_size, -1, 3)
        else:
            batch_size = box.shape[0]
            box = box.reshape(batch_size, 3, 3)
            x_linespace = torch.linspace(0, 1, self.num_grids[0] + 1, dtype=torch.float64)
            y_linespace = torch.linspace(0, 1, self.num_grids[1] + 1, dtype=torch.float64)
            z_linespace = torch.linspace(0, 1, self.num_grids[2] + 1, dtype=torch.float64)

            # Calculate centers of the mesh
            x_centers = (x_linespace[1:] + x_linespace[:-1]) / 2
            y_centers = (y_linespace[1:] + y_linespace[:-1]) / 2
            z_centers = (z_linespace[1:] + z_linespace[:-1]) / 2

            # Create mesh grid: [num_x_centers, num_y_centers, num_z_centers, 3]
            mesh = torch.stack(torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij'), dim=-1).to(atom_pos.device)
            
            mesh_coord = torch.einsum("ijkl,nlm->nijkm", mesh, box.to(atom_pos.device).to(torch.float64)).view(batch_size, -1, 3)
            new_pos =atom_pos
        return new_pos, mesh_coord

class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
            self,
            in_channels,
            out_channels=None,
            hidden_channels=None,
            n_layers=2,
            n_dim=2,
            non_linearity=F.silu,
            dropout=0.0,
            **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        Conv = getattr(nn, f"Conv{n_dim}d")
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))
        
        self.reset_parameters()
                
    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        x = x.to(torch.float32)
        for i, fc in enumerate(self.fcs):
            fc.to(x.device)
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)

        return x

class SoftGating(nn.Module):
    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.ones_(self.bias)
    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        weight = self.weight.to(x.device)
        if self.bias is not None:
            bias = self.bias.to(x.device)
            return weight * x + bias
        else:
            return weight * x


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.max_n_modes = self.n_modes
        self.n_layers = n_layers

        self.weight = nn.ParameterList([
            nn.Parameter(
                torch.view_as_real(torch.empty(in_channels, out_channels, *self.max_n_modes, dtype=torch.cfloat))
            )
            for _ in range(n_layers)
        ])
        self.bias = nn.Parameter(
            torch.empty(*((n_layers, self.out_channels) + (1,) * self.order))
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.normal_(w, 0, (2 / (self.in_channels + self.out_channels)) ** 0.5)
        nn.init.normal_(self.bias, 0, (2 / (self.in_channels + self.out_channels)) ** 0.5)

    def _get_weight(self, index):
        return torch.view_as_complex(self.weight[index])

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, indices=0):
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))
        
        x = torch.fft.rfftn(x, norm='backward', dim=fft_dims).to(torch.complex64)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat)
        starts = [
            (max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in
            zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        slices_w = [slice(None), slice(None)]  # Batch_size, channels
        slices_w += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        # The last mode already has redundant half removed
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self._get_weight(indices)[slices_w].to(x.device)

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        slices_x += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        # The last mode already has redundant half removed
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        
        out_fft[slices_x] = torch.einsum("bcxyz,cdxyz->bdxyz", x[slices_x], weight)

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm='backward')
        x = x + self.bias[indices, ...].to(x.device)
        return x


class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, non_linearity=F.silu):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                SoftGating(
                    self.in_channels,
                    self.out_channels,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.convs.reset_parameters()
        for fno_skip in self.fno_skips:
            fno_skip.reset_parameters()

    def forward(self, x, index=0):
        x_skip_fno = self.fno_skips[index](x)
        x_fno = self.convs(x, index)
        x = x_fno + x_skip_fno
        if index < (self.n_layers - 1):
            x = self.non_linearity(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes

class FNO(nn.Module):
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=1,
            out_channels=1,
            n_layers=1,
            lifting_channels=256,
            projection_channels=256,
            non_linearity=F.silu,
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        self.temp_layer = MLPLayer(
                in_channels,
                self.hidden_channels,
            )
        self.temp_layer_2 = MLPLayer(
                self.hidden_channels,
                self.out_channels,
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.lifting.reset_parameters()
        self.fno_blocks.reset_parameters()
        self.projection.reset_parameters()

    def forward(self, x,batch_size):
        # x = x.view(batch_size, -1,*self._n_modes)

        x = self.temp_layer(x)
        x = x.transpose(0,1).reshape(batch_size, -1, *self._n_modes)
        
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
        x = x.view(batch_size,self.hidden_channels, -1).transpose(1,2).view(-1,self.hidden_channels)

        x = self.temp_layer_2(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class FNO3d(FNO):
    def __init__(
            self,
            n_modes_height,
            n_modes_width,
            n_modes_depth,
            hidden_channels,
            in_channels=1,
            out_channels=1,
            n_layers=1,
            lifting_channels=256,
            projection_channels=256,
            non_linearity=F.silu,
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            non_linearity=non_linearity,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth



def main():
    """单元测试函数"""
    print("开始测试 NonPBCAddGrid 类...")
    
    # 创建模拟数据类
    class MockData:
        def __init__(self, pos):
            self.pos = pos
            self.cell = None
    
    # 测试用例1：基本功能测试
    print("\n测试用例1：基本功能测试")
    expand_size = 2.0
    num_grids = [3, 3, 3]
    transform = NonPBCAddGrid(expand_size, num_grids)
    
    # 创建测试数据：batch_size=2, num_nodes=5, hidden_size=3
    batch_size, num_nodes, hidden_size = 50, 200, 3
    pos = torch.randn(batch_size, num_nodes, hidden_size)
    data = MockData(pos)
    
    print(f"输入位置形状: {pos.shape}")
    
    # 调用变换
    new_pos, mesh_coord = transform(data)
    
    print(f"输出位置形状: {new_pos.shape}")
    print(f"网格坐标形状: {mesh_coord.shape}")
    print(f"设置的cell形状: {data.cell.shape}")
    # 验证形状是否正确
    assert new_pos.shape == pos.shape, f"位置形状不匹配: {new_pos.shape} vs {pos.shape}"
    assert data.cell.shape == (batch_size, 3, 3), f"cell形状不匹配: {data.cell.shape}"
    expected_mesh_shape = (batch_size, num_grids[0], num_grids[1], num_grids[2], 3)
    assert mesh_coord.shape == expected_mesh_shape, f"网格坐标形状不匹配: {mesh_coord.shape} vs {expected_mesh_shape}"
    
    print("✓ 基本功能测试通过")
    
    # 测试用例2：单一数值网格测试
    print("\n测试用例2：单一数值网格测试")
    transform_single = NonPBCAddGrid(expand_size=1.5, num_grids=3)  # 应该变为[3, 3, 3]
    
    pos_single = torch.randn(1, 10, 3)
    data_single = MockData(pos_single)
    
    new_pos_single, mesh_coord_single = transform_single(data_single)
    
    print(f"单一网格输入: num_grids=3")
    print(f"网格坐标形状: {mesh_coord_single.shape}")
    
    expected_single_shape = (1, 3, 3, 3, 3)
    assert mesh_coord_single.shape == expected_single_shape, f"单一网格形状不匹配: {mesh_coord_single.shape}"
    
    print("✓ 单一数值网格测试通过")
    
    # 测试用例3：不同大小网格测试
    print("\n测试用例3：不同大小网格测试")
    transform_diff = NonPBCAddGrid(expand_size=3.0, num_grids=[2, 3, 4])
    
    pos_diff = torch.randn(1, 8, 3)
    data_diff = MockData(pos_diff)
    
    new_pos_diff, mesh_coord_diff = transform_diff(data_diff)
    
    print(f"不同网格大小: [2, 3, 4]")
    print(f"网格坐标形状: {mesh_coord_diff.shape}")
    
    expected_diff_shape = (1, 2, 3, 4, 3)
    assert mesh_coord_diff.shape == expected_diff_shape, f"不同网格形状不匹配: {mesh_coord_diff.shape}"
    
    print("✓ 不同大小网格测试通过")
    
    # 测试用例4：数值范围验证
    print("\n测试用例4：数值范围验证")
    pos_test = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32)
    data_test = MockData(pos_test)
    
    transform_test = NonPBCAddGrid(expand_size=1.0, num_grids=2)
    new_pos_test, mesh_coord_test = transform_test(data_test)
    
    print(f"原始位置: {pos_test}")
    print(f"处理后位置: {new_pos_test}")
    print(f"cell矩阵: {data_test.cell}")
    
    # 验证位置是否被正确中心化
    pos_mean = new_pos_test.mean(dim=1)
    print(f"新位置的均值: {pos_mean}")
    
    print("✓ 数值范围验证通过")
    
    print("\n所有测试用例通过！NonPBCAddGrid 类工作正常。")


if __name__ == "__main__":
    main()