# 单线程模拟 4-way Graph Parallel 验证指南

## 快速开始

### 1. 运行基础测试

```bash
cd /aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-arch
python test_gp_single_thread.py
```

这个测试会验证：
- Index Remapping 的正确性
- GP 模式是否正常激活
- 节点分区是否正确

### 2. 运行完整训练测试

```bash
cd /aisi/mnt/data_nas/liwentao/devel_workspace/deepmd-kit-arch/test_mptraj
bash run.sh
```

或者直接运行：

```bash
dp --pt train --skip-neighbor-stat test_mptraj/input_dynamic.json
```

---

## 验证检查点

### 检查点 1: repflows.py 是否进入 GP 模式

**位置**: `deepmd/pt/model/descriptor/repflows.py:810`

**添加调试代码**:

```python
if gp_enabled and self.use_dynamic_sel:
    if not is_debug:
        print(f"✓ GP 模式激活: world_size={gp_world_size}")
        print(f"  节点总数: {gp_num_nodes}")
        print(f"  分区大小: {gp_sizes}")
```

**预期输出**:

```
✓ GP 模式激活: world_size=4
  节点总数: 100
  分区大小: [25, 25, 25, 25]
```

---

### 检查点 2: 每个分区的 Encoder 计算

**位置**: `deepmd/pt/model/descriptor/repflows.py:820`

**添加调试代码**:

```python
for partition in gp_partitions:
    rank = partition['rank']
    local_start = partition['local_start']
    local_end = partition['local_end']
    local_size = partition['local_size']

    print(f"  Rank {rank}: nodes [{local_start}, {local_end}), size={local_size}")
    print(f"    edges: {partition['edge_index'].shape[1]}")
    print(f"    angles: {partition['angle_index'].shape[1]}")
```

**预期输出**:

```
  Rank 0: nodes [0, 25), size=25
    edges: 500
    angles: 150
  Rank 1: nodes [25, 50), size=25
    edges: 480
    angles: 145
  Rank 2: nodes [50, 75), size=25
    edges: 490
    angles: 148
  Rank 3: nodes [75, 100), size=25
    edges: 510
    angles: 152
```

---

### 检查点 3: h2g2 计算的局部化

**位置**: `deepmd/pt/model/descriptor/repflows.py:910`

**添加调试代码**:

```python
for i, partition in enumerate(gp_partitions):
    local_owner = local_edge_index[0] - local_start

    print(f"  Rank {i}:")
    print(f"    local_owner min/max: {local_owner.min().item()}/{local_owner.max().item()}")
    print(f"    expected range: [0, {local_size})")

    # 验证 owner 在正确范围内
    assert local_owner.min() >= 0, f"Rank {i}: owner < 0"
    assert local_owner.max() < local_size, f"Rank {i}: owner >= local_size"
```

**预期输出**:

```
  Rank 0:
    local_owner min/max: 0/24
    expected range: [0, 25)
  Rank 1:
    local_owner min/max: 0/24
    expected range: [0, 25)
  ...
```

---

### 检查点 4: Fitting Net 处理局部节点

**位置**: `deepmd/pt/model/atomic_model/dp_atomic_model.py:280`

**添加调试代码**:

```python
if isinstance(descriptor_output, dict) and 'gp_partitions' in descriptor_output:
    print(f"✓ Fitting Net 进入 GP 模式")

    for i, partition in enumerate(gp_partitions):
        local_descriptor = node_ebd_parts[i]
        local_atype = atype[:, partition['local_start']:partition['local_end']]

        print(f"  Rank {i}:")
        print(f"    descriptor shape: {local_descriptor.shape}")
        print(f"    atype shape: {local_atype.shape}")
```

**预期输出**:

```
✓ Fitting Net 进入 GP 模式
  Rank 0:
    descriptor shape: torch.Size([1, 25, 128])
    atype shape: torch.Size([1, 25])
  Rank 1:
    descriptor shape: torch.Size([1, 25, 128])
    atype shape: torch.Size([1, 25])
  ...
```

---

### 检查点 5: 能量 All-Reduce

**位置**: `deepmd/pt/model/model/transform_output.py:260`

**添加调试代码**:

```python
if 'gp_mode' in fit_ret and fit_ret['gp_mode']:
    print(f"✓ 能量 All-Reduce")

    # 打印每个分区的能量
    for i, part in enumerate(model_ret_parts):
        local_energy = part['energy_redu'].item()
        print(f"  Rank {i} energy: {local_energy:.6f}")

    # 打印总能量
    total_energy = model_ret['energy_redu'].item()
    print(f"  Total energy (GP): {total_energy:.6f}")

    # 验证：总能量 = 各分区能量之和
    sum_energy = sum([part['energy_redu'].item() for part in model_ret_parts])
    print(f"  Sum of parts: {sum_energy:.6f}")
    print(f"  Difference: {abs(total_energy - sum_energy):.10f}")
```

**预期输出**:

```
✓ 能量 All-Reduce
  Rank 0 energy: -125.342156
  Rank 1 energy: -118.765432
  Rank 2 energy: -122.987654
  Rank 3 energy: -120.123456
  Total energy (GP): -487.218698
  Sum of parts: -487.218698
  Difference: 0.0000000000
```

---

## 对比验证：GP vs 全图

### 步骤 1: 运行 GP 模式

```python
# repflows.py Line 697
is_debug = False  # 启用 GP 模式
```

运行并记录结果：

```bash
dp --pt train --skip-neighbor-stat test_mptraj/input_dynamic.json > gp_output.log 2>&1
```

提取能量：

```bash
grep "Total energy" gp_output.log | head -10
```

### 步骤 2: 运行全图模式

```python
# repflows.py Line 697
is_debug = True  # 禁用 GP 模式，使用全图计算
```

运行并记录结果：

```bash
dp --pt train --skip-neighbor-stat test_mptraj/input_dynamic.json > full_output.log 2>&1
```

提取能量：

```bash
grep "Total energy" full_output.log | head -10
```

### 步骤 3: 对比结果

```python
import numpy as np

# 读取 GP 模式的能量
gp_energies = [...]  # 从 gp_output.log 提取

# 读取全图模式的能量
full_energies = [...]  # 从 full_output.log 提取

# 计算差异
diff = np.abs(np.array(gp_energies) - np.array(full_energies))
print(f"Max difference: {diff.max():.10f}")
print(f"Mean difference: {diff.mean():.10f}")

# 验证
if diff.max() < 1e-5:
    print("✓ GP 模式和全图模式结果一致！")
else:
    print("✗ GP 模式和全图模式结果不一致！")
```

---

## 常见问题排查

### 问题 1: IndexError: index out of range

**症状**:

```
IndexError: index 150 is out of bounds for dimension 0 with size 100
```

**原因**: Index Remapping 失败，angle_index 仍然使用全局边索引。

**排查**:

```python
# 在 repflows.py:733 添加验证
print(f"Before remap: angle_index[1] = {gp_angle_index[1][:5]}")
gp_angle_index[1] = edge_remap[gp_angle_index[1]]
print(f"After remap: angle_index[1] = {gp_angle_index[1][:5]}")

# 验证范围
assert gp_angle_index[1].min() >= 0
assert gp_angle_index[1].max() < gp_edge_index.shape[1]
```

---

### 问题 2: 能量不一致

**症状**: GP 模式和全图模式的能量差异 > 1e-5

**原因**:
1. h2g2 计算的 owner 索引错误
2. 分区边界处理不正确
3. 梯度计算错误

**排查**:

```python
# 1. 验证 owner 范围
local_owner = local_edge_index[0] - local_start
assert local_owner.min() >= 0
assert local_owner.max() < local_size

# 2. 验证节点总数
total_nodes = sum([part.shape[1] for part in node_ebd_parts])
assert total_nodes == nloc

# 3. 验证能量求和
sum_energy = sum([part['energy_redu'].item() for part in model_ret_parts])
assert abs(total_energy - sum_energy) < 1e-10
```

---

### 问题 3: 梯度为 None

**症状**:

```
RuntimeError: grad can be implicitly created only for scalar outputs
```

**原因**: 局部能量不是标量，无法直接求导。

**解决**: 确保每个分区的能量已经求和为标量：

```python
# transform_output.py:240
local_model_ret[kk_redu] = torch.sum(vv.to(redu_prec), dim=atom_axis)
# 确保这是标量
assert local_model_ret[kk_redu].dim() == 0 or local_model_ret[kk_redu].numel() == 1
```

---

## 性能分析

### 预期性能提升

假设：
- 节点数 N = 1000
- 边数 E = 50000
- GP world_size = 4

**显存占用**:

| 阶段 | 全图模式 | GP 模式 | 节省 |
|------|---------|---------|------|
| Encoder | N × n_dim | N × n_dim | 0% (需要全量节点) |
| Fitting | N × n_dim | (N/4) × n_dim | 75% |
| 总计 | ~100% | ~40% | 60% |

**通信开销**:

| 操作 | 数据量 | 频率 |
|------|--------|------|
| All-Gather (Encoder) | N × n_dim | 每层 (11次) |
| All-Reduce (Energy) | 1 标量 | 1次 |

**理论加速比**: 1.5x - 2x (取决于通信/计算比)

---

## 下一步：真实分布式

当单线程模拟验证通过后，修改为真实分布式：

### 修改 1: 使用真实的 rank 和 world_size

```python
# repflows.py Line 700-703
if gp_enabled and self.use_dynamic_sel and not is_debug:
    # ✅ 使用真实的分布式信息
    import distutils
    gp_rank = distutils.get_gp_rank()
    gp_world_size = distutils.get_gp_world_size()

    # 只计算当前 rank 的分区
    gp_num_nodes = node_ebd.shape[0]
    gp_sizes = graph_parallel._balanced_partition_sizes(gp_num_nodes, gp_world_size)
    gp_offsets = graph_parallel._build_partition_offsets(gp_sizes, node_ebd.device)
    gp_local_start = int(gp_offsets[gp_rank].item())
    gp_local_end = gp_local_start + gp_sizes[gp_rank]

    # 只处理当前 rank 的边和角
    gp_edge_mask = (edge_index[0] >= gp_local_start) & (edge_index[0] < gp_local_end)
    gp_edge_index = edge_index[:, gp_edge_mask]
    # ...
```

### 修改 2: 使用真实的 All-Gather

```python
# repflows.py Line 860
if not is_last_layer:
    # ✅ 使用真实的 All-Gather
    import distutils
    node_ebd = distutils.gather_from_model_parallel_region_sum_grad(node_ebd_local, dim=0)
else:
    node_ebd = node_ebd_local
```

### 修改 3: 使用真实的 All-Reduce

```python
# transform_output.py Line 260
# ✅ 使用真实的 All-Reduce
import distutils
model_ret[kk_redu] = distutils.reduce_from_model_parallel_region(local_energy)
```

---

## 总结

本验证指南提供了：

1. ✅ 5 个关键检查点，确保每个阶段正确
2. ✅ 对比验证方法，确保 GP 模式和全图模式一致
3. ✅ 常见问题排查，快速定位错误
4. ✅ 性能分析，评估优化效果
5. ✅ 真实分布式迁移指南

按照本指南逐步验证，可以确保单线程模拟的正确性，为真实分布式实现奠定基础。
