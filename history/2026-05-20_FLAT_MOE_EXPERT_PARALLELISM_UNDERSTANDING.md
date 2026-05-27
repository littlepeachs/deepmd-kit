# 当前 flat 版本 MoE 专家并行实现学习记录

日期: 2026-05-20

仓库: `/aisi-nas/liwentao/deepmd-kit-moe`

关注点: 当前 `flat` mixed-batch 路径下, DPA3/RepFlow 的 MoE 专家并行如何被调用、如何路由、如何通信、如何回到能量/力/virial/loss。

## 总结

当前 flat 版本不是重新实现了一套 MoE, 而是把 LMDB mixed-nloc batch 转成一套 flattened atom graph, 再把这套 flat node/edge/angle token 输入到已有的 RepFlowLayer MoE 专家并行路径中。

核心设计可以概括为:

```text
LMDB mixed batch
  -> concat atom-wise tensors
  -> build flat graph on DEVICE
  -> model forward detects batch/ptr
  -> DPA3.forward_flat
  -> RepFlows.forward_flat
  -> RepFlowLayer.forward_moe
  -> MoEDispatchCombine expert parallel all-to-all
  -> ordinary atomic energy / force / virial / loss
```

flat 的作用是解决 mixed-nloc batch 不能直接表示成 `[nframes, nloc, ...]` dense tensor 的问题。MoE EP 的作用是把 routing expert 切到多个 GPU 上, token 根据 router 结果发到对应 GPU 上计算, 再 all-to-all 发回原 rank。

两者的结合点在 `RepFlows.forward_flat()`:

- mixed batch 在 descriptor 前已经变成 flat token。
- flat path 把所有原子临时包装成一个 synthetic frame: `[1, total_atoms, dim]`。
- graph/nlist/edge_index/angle_index 保证真实 frame 之间不会连边。
- MoE layer 仍然看到标准的 node/edge/angle token, 因此复用同一套 `forward_moe()` 和 `MoEDispatchCombine`。

## 典型使用方式

配置里需要同时满足:

```json
{
  "training": {
    "training_data": {
      "systems": "/path/to/data.lmdb",
      "batch_size": 2,
      "mixed_batch": true
    },
    "moe_ep_size": 8
  },
  "model": {
    "descriptor": {
      "type": "dpa3",
      "repflow": {
        "use_moe": true
      }
    }
  }
}
```

8 卡启动命令:

```bash
conda activate /aisi-nas/liwentao/miniconda/dpa3_moe
torchrun --standalone --nproc_per_node=8 "$(which dp)" --pt train --skip-neighbor-stat input.json
```

对于 8 卡且 `moe_ep_size=8`:

- `world_size = 8`
- `ep_size = 8`
- `dp_size = 1`
- 8 个 rank 在同一个 EP group 里做 MoE token all-to-all
- 每张卡只保存一部分 routing experts

## 入口和 EP/DP group 初始化

入口在 `deepmd/pt/entrypoints/main.py::get_trainer()`。

它在 distributed 初始化后判断是否启用 MoE EP:

```text
dist initialized
  descriptor.type == "dpa3"
  repflow.use_moe == true
  training.moe_ep_size > 1
    -> init_ep_dp_groups(moe_ep_size)
    -> use_moe_ep = True
```

group 划分由 `deepmd/pt/utils/moe_ep_dp.py::init_ep_dp_groups()` 完成。它把全局 GPU 看成一个二维网格:

```text
world_size = ep_size * dp_size

ep_size=2, dp_size=2:

              EP rank 0  EP rank 1
DP rank 0:    GPU 0      GPU 1      <- ep_group_0
DP rank 1:    GPU 2      GPU 3      <- ep_group_1
              ^          ^
              dp_group_0 dp_group_1
```

对于用户当前的 8 卡命令, 如果 `moe_ep_size=8`, 网格是:

```text
DP rank 0: GPU 0 GPU 1 GPU 2 GPU 3 GPU 4 GPU 5 GPU 6 GPU 7
           EP0   EP1   EP2   EP3   EP4   EP5   EP6   EP7
```

这意味着:

- `ep_group`: 8 张卡全部参与 MoE dispatch/combine。
- `dp_group`: 每列只有 1 张卡, routing expert 没有额外 DP 副本。
- 非 routing 参数仍然在所有 rank 上各有一份, 训练时需要同步梯度。

## MoE 上下文如何进入模型

`Trainer.__init__()` 在构建模型前做:

```python
set_moe_ep_context(self.ep_group, self.ep_rank, self.ep_size)
self.model = get_model_for_wrapper(...)
```

实现位置:

- `deepmd/pt/train/training.py`
- `deepmd/pt/utils/moe_context.py`

原因是 `ProcessGroup` 不能直接放进 JSON config, 否则 deepcopy 和序列化都会有问题。所以这里用 thread-local context 注入。

随后 `DPA3.__init__()` 从 thread-local 读取 EP 参数, 再把它传给 `RepFlows` 和每个 `RepFlowLayer`。最终每层 MoE 会知道:

- `ep_group`
- `ep_rank`
- `ep_size`
- `experts_per_gpu = n_routing_experts // ep_size`

如果 `n_routing_experts` 不能被 `ep_size` 整除, MoE layer 构造阶段会报错。

## LMDB mixed batch 如何变成 flat batch

入口在 `deepmd/pt/entrypoints/main.py::prepare_trainer_input_single()`。

如果 `training.training_data.systems` 是 LMDB 路径, 会构造:

```python
LmdbDataset(
    training_systems,
    type_map,
    batch_size,
    mixed_batch=mixed_batch,
    auto_prob_style=auto_prob,
)
```

当 `mixed_batch=True` 时, `Trainer.get_data_loader()` 用:

```python
collate_fn = _collate_lmdb_mixed_batch
```

当前实现有一个重要点: collate 阶段只做 CPU 上的拼 batch, 不在 collate 里构图。flat graph 会在 `Trainer.get_data()` 中, batch tensor 移到 `DEVICE` 之后再构建。

这样设计的原因是 dense 路径也在 GPU 上构 neighbor list。CPU 和 GPU 在等距周期镜像、`topk` tie-break 等情况下可能选出不同邻居顺序。如果 flat graph 在 CPU 上提前构, same-nloc 等价测试可能出现能量或力的微小不一致。当前 flat graph 在 DEVICE 上构建, 与 dense 路径保持一致。

## `_collate_lmdb_mixed_batch()` 的输出

位置: `deepmd/pt/utils/lmdb_dataset.py`

它把一个 list of frame dict 拼成一个 batch dict。规则是:

- atom-wise 字段直接 concat。
- frame-wise 字段 stack。
- 附加 `batch` 和 `ptr` 表示 frame 边界。

atom-wise 字段包括:

```text
coord, atype, force, aparam, atom_ener, atom_pref,
atomic_weight, spin, hessian, drdq, ...
```

frame-wise 字段包括:

```text
box, energy, virial, fparam, ...
```

举例, 假设一个 batch 有两帧:

```text
frame 0: nloc = 4
frame 1: nloc = 7
```

collate 后:

```text
coord: [11, 3]
atype: [11]
force: [11, 3]
energy: [2, 1]
virial: [2, 9]
batch: [0,0,0,0, 1,1,1,1,1,1,1]
ptr: [0, 4, 11]
```

这里:

- `batch[i]` 表示第 `i` 个 flat atom 属于哪一帧。
- `ptr[f] : ptr[f+1]` 是第 `f` 帧在 flat atom tensor 里的区间。

## flat graph 在哪里构建

位置: `deepmd/pt/train/training.py::get_data()`。

`get_data()` 拿到 batch 后:

1. 判断 `is_mixed_batch = "batch" in batch_data and "ptr" in batch_data`。
2. 把 `coord/atype/box/force/...` 移到 `DEVICE`。
3. 把 `batch/ptr` 也移到 `DEVICE`。
4. 如果还没有 `nlist`, 调用 `build_precomputed_flat_graph()`。

graph 参数来自模型 descriptor:

```python
graph_config = {
    "rcut": descriptor.get_rcut(),
    "sel": descriptor.get_sel(),
    "a_rcut": descriptor.repflows.a_rcut,
    "a_sel": descriptor.repflows.a_sel,
    "mixed_types": descriptor.mixed_types(),
}
```

因此当前 `mixed_batch=True` 要求 descriptor 是 flat-graph capable 的 DPA3/RepFlow。普通 descriptor 没有 `repflows` 时会直接报错。

## `build_precomputed_flat_graph()` 的职责

位置: `deepmd/pt/utils/nlist.py::build_precomputed_flat_graph()`

输入:

```text
coord: [total_atoms, 3]
atype: [total_atoms]
batch: [total_atoms]
ptr: [nframes + 1]
box: [nframes, 9] or None
rcut, sel, a_rcut, a_sel, mixed_types
```

输出是一组 flat graph tensor:

```text
extended_atype      [total_extended_atoms]
extended_batch      [total_extended_atoms]
extended_image      [total_extended_atoms, 3]
extended_ptr        [nframes + 1]
mapping             [total_extended_atoms]
central_ext_index   [total_atoms]
nlist               [total_atoms, nnei]
nlist_ext           [total_atoms, nnei]
a_nlist             [total_atoms, a_sel]
a_nlist_ext         [total_atoms, a_sel]
nlist_mask          [total_atoms, nnei]
a_nlist_mask        [total_atoms, a_sel]
edge_index          [2, n_edge]
angle_index         [3, n_angle]
```

几个字段的含义非常关键:

- `extended_atype`: ghost 扩展后的原子类型。
- `extended_batch`: 每个 extended atom 属于哪一帧。
- `extended_image`: ghost image 的晶胞平移整数, 用于后续从 `coord` 和 `box` 可导地重建 extended 坐标。
- `mapping`: extended atom 映射回 flat local atom index。
- `central_ext_index`: 每个真实 atom 在 extended atom tensor 中的 index。
- `nlist_ext`: 邻居在 extended atom tensor 中的 index, 用来取坐标和算距离。
- `nlist`: 邻居映射回 flat local atom index, 用来复用真实原子的 node embedding。
- `a_nlist_ext` / `a_nlist`: angle neighbor 对应的 extended/local 两套 index。
- `edge_index`: dynamic edge 的 owner atom 和 neighbor atom index。
- `angle_index`: dynamic angle 的 owner atom, edge j, edge k index。

这里同时保留 `*_ext` 和非 `*_ext` 是 flat path 的关键。距离必须在 extended coordinate 空间算, 否则 PBC ghost 会错；node embedding 又应该来自原始真实原子, 不能给 ghost 单独创建新 token。

## same-nloc 和 mixed-nloc 两条构图分支

`build_precomputed_flat_graph()` 首先检查:

```python
atom_counts = ptr[1:] - ptr[:-1]
if torch.all(atom_counts == atom_counts[0]):
    # same-nloc fast/equivalence branch
else:
    # real mixed-nloc branch
```

same-nloc 分支会把 flat tensor reshape 成 dense:

```text
coord [total_atoms, 3] -> [nframes, nloc, 3]
atype [total_atoms]    -> [nframes, nloc]
```

然后用和 dense 路径相同的批量 neighbor list 构建方式, 最后再 flatten 回来。

这个分支的目的不是加速本身, 而是验证等价性: 当所有帧 nloc 一样时, flat 数据处理方式和原 dense 数据处理方式应该给出一致结果。

mixed-nloc 分支则逐帧循环:

```text
for frame_idx in range(nframes):
    frame_coord = coord[ptr[i]:ptr[i+1]]
    extend ghosts
    build frame nlist
    add extended_offset / local atom offset
concat all frame graph tensors
```

无论哪条分支, 最终得到的都是一套 batch 级 flat graph。真实 frame 之间不会互相连边。

## extended coordinate 为什么 forward 时重建

flat graph 中保存的是:

- `mapping`
- `extended_batch`
- `extended_image`

forward 时在 `forward_common_flat_native()` 里调用:

```python
rebuild_extended_coord_from_flat_graph(
    coord,
    box,
    mapping,
    extended_batch,
    extended_image,
)
```

这样做的原因是 force/virial 需要对输入 `coord` 和 `box` 求导。如果 collate 阶段直接保存一个 detached 的 `extended_coord`, 梯度链路会断掉。

重建逻辑本质是:

```text
extended_coord = coord[mapping] + extended_image @ box[extended_batch]
```

因此:

- force 可以从 `energy` 对原始 flat `coord` 求导。
- virial 可以从 extended force 和 extended coord 聚合得到。
- PBC ghost 的贡献仍然能回到原始 atom。

## flat 输入如何进入 ModelWrapper

位置: `deepmd/pt/train/wrapper.py::ModelWrapper.forward()`

普通输入:

```text
coord, atype, box, fparam, aparam
```

如果 `batch is not None and ptr is not None`, wrapper 额外传入:

```text
batch, ptr,
extended_atype, extended_batch, extended_image, extended_ptr,
mapping, central_ext_index,
nlist, nlist_ext,
a_nlist, a_nlist_ext,
nlist_mask, a_nlist_mask,
edge_index, angle_index
```

loss 调用时 `natoms` 也做了适配:

```python
natoms = atype.shape[-1] if atype.dim() > 1 else atype.shape[0]
```

对于 flat batch, `atype.dim()==1`, 所以 `natoms=total_atoms`。

## 模型如何选择 flat forward

位置: `deepmd/pt/model/model/ener_model.py`

`EnergyModel.forward()` 中只要看到:

```python
batch is not None and ptr is not None
```

就进入:

```text
forward_common_flat()
  -> forward_common_flat_native()
  -> forward_common_lower_flat()
  -> atomic_model.forward_common_atomic_flat()
```

这条路径区别于 dense forward。dense forward 会先 build/communicate extended input, 再 `[nframes, nloc, ...]` 处理；flat forward 已经有预计算 flat graph, 直接使用。

## `forward_common_flat_native()` 做了什么

位置: `deepmd/pt/model/model/make_model.py`

主要步骤:

1. 输入 dtype cast。
2. 如果需要 force, 设置 `coord.requires_grad_(True)`。
3. 如果需要 virial 且有 `box`, 设置 `box.requires_grad_(True)`。
4. 用 `mapping/extended_batch/extended_image` 重建 `extended_coord`。
5. 调 `forward_common_lower_flat()`。
6. 如果需要 force/virial, 调 `_compute_derivatives_flat()`。

如果 flat graph 字段缺失, 会直接报错:

```text
Flat mixed-batch forward requires precomputed graph fields
```

这意味着 flat forward 不会在模型内部临时补构图；构图职责明确放在 data 阶段。

## flat lower forward 如何 reduce energy

`forward_common_lower_flat()` 调 atomic model 得到 atom-wise 输出后, 对能量做 frame reduce:

```python
energy_redu = energy_atomic.new_zeros((nframes, energy_atomic.shape[-1]))
energy_redu.index_add_(0, batch, energy_atomic)
```

含义是:

```text
energy_redu[f] = sum energy_atomic[i] for batch[i] == f
```

这避免了把 atom energy padding 到 `[nframes, max_nloc]` 后再 reduce。对 mixed-nloc batch, `batch` 就是 frame 归属的唯一标准。

## flat atomic model

位置: `deepmd/pt/model/atomic_model/dp_atomic_model.py::forward_common_atomic_flat()`

调用链:

```text
forward_common_atomic_flat
  -> descriptor.forward_flat(...)
  -> fitting_net.forward_flat(...)
  -> apply_out_stat(...)
```

这里有两个重要点:

1. 如果需要 force/virial, `extended_coord.requires_grad_(True)`。
2. descriptor 走真正的 flat graph path, fitting net 则暂时复用 dense fitting path。

## DPA3 flat descriptor

位置: `deepmd/pt/model/descriptor/dpa3.py::forward_flat()`

DPA3 的 flat descriptor 做:

1. `node_ebd_ext = self.type_embedding(extended_atype)`。
2. 如果有 charge/spin embedding, 根据 `extended_batch` 把 frame 级 `fparam` 映射到 extended atom。
3. 用 `central_ext_index` 取真实 atom 的初始 embedding:

```python
node_ebd_inp = node_ebd_ext[central_ext_index]
```

4. 调 `self.repflows.forward_flat(...)`。
5. 如果 `concat_output_tebd`, 把输出 descriptor 和初始 type embedding concat。

DPA3 本身不直接做 MoE dispatch。MoE 在 RepFlowLayer 内部发生。

## RepFlows.forward_flat 的核心逻辑

位置: `deepmd/pt/model/descriptor/repflows.py::forward_flat()`

输入已经是 flat graph。它首先用 `prod_env_mat_flat()` 计算 edge environment:

```text
dmatrix, diff, sw = prod_env_mat_flat(
    extended_coord,
    nlist_ext,
    atype,
    mean,
    stddev,
    e_rcut,
    ...
    coord_flat=coord_central,
)
```

这里必须用 `nlist_ext`, 因为距离要在 extended atom 空间取邻居坐标。

angle 部分同理:

```text
a_diff, a_sw = prod_env_mat_flat(..., a_nlist_ext, ...)
```

如果 `use_dynamic_sel=True`, 会用 mask 把 dense neighbor slot 压成真正的 dynamic edge/angle token:

```python
edge_input = edge_input[nlist_mask]
h2 = h2[nlist_mask]
sw = sw[nlist_mask]

a_nlist_mask_2d = a_nlist_mask[:, :, None] & a_nlist_mask[:, None, :]
angle_input = angle_input[a_nlist_mask_2d]
a_sw = (a_sw[:, :, None] * a_sw[:, None, :])[a_nlist_mask_2d]
```

随后构造初始 embedding:

```text
node_ebd:  [total_atoms, n_dim]
edge_ebd:  [n_edge, e_dim]
angle_ebd: [n_angle, a_dim]
```

为了复用原 RepFlowLayer 接口, flat path 包一层 synthetic batch:

```python
node_ebd_batched = node_ebd.unsqueeze(0)      # [1, total_atoms, n_dim]
nlist_batched = nlist.unsqueeze(0)           # [1, total_atoms, nnei]
a_nlist_batched = a_nlist.unsqueeze(0)       # [1, total_atoms, a_sel]
atype_embd_batched = atype_embd.unsqueeze(0) # [1, total_atoms, n_dim]
```

然后每层:

```python
ll.forward(
    node_ebd_ext_batched,
    edge_ebd_batched,
    h2_batched,
    angle_ebd_batched,
    nlist_batched,
    nlist_mask_batched,
    sw_batched,
    a_nlist_batched,
    a_nlist_mask_batched,
    a_sw_batched,
    edge_index=edge_index,
    angle_index=angle_index,
    type_embedding=atype_embd_batched if self.use_moe else None,
)
```

这里 `type_embedding` 对 MoE 非常关键。router 用它决定每个 node token 应该进哪些专家。flat path 必须传这个参数, 否则 MoE router 没有输入。

## RepFlowLayer 什么时候进入 MoE

位置: `deepmd/pt/model/descriptor/repflow_layer.py`

`RepFlowLayer.forward()` 中如果 `self.use_moe=True`, 会要求 `type_embedding` 非空, 然后进入:

```text
forward_moe(...)
```

当前 MoE 有几个约束:

- `use_dynamic_sel=True`
- `optim_update=False`
- `update_angle=True`
- `n_multi_edge_message=1`
- `a_compress_use_split=True`

这些约束来自 MoE packer 和合并 MLP 的实现假设。当前 packer 依赖固定维度比例:

```text
n_dim : e_dim : a_dim = 4 : 2 : 1
```

因此不是任意 RepFlow 配置都能打开 MoE。

## RepFlowLayer.forward_moe 的 token 输入

`forward_moe()` 把一个 RepFlowLayer 的 Phase 1 MLP 输入拆成四类 expert token:

### node M1

```text
node_m1_input = node_ebd.reshape(N_node, n_dim)
```

对应原 M1, 输入/输出:

```text
n_dim -> n_dim
```

### node M2

先做 dynamic symmetrization:

```text
grrg = sym(edge_ebd, h2, sw)
drrd = sym(nei_node_ebd, h2, sw)
node_m2_input = cat(grrg, drrd)
```

输入/输出:

```text
n_sym_dim -> n_dim
```

### edge merged expert

把 M3 和 M4 合成一个 expert:

```text
edge_info = cat(node_i, node_j, edge_ebd)
```

输入/输出:

```text
edge_info_dim -> n_dim + e_dim
```

输出后再 split:

```text
node_edge_out, edge_self_out = edge_merged_out.split([n_dim, e_dim])
```

### angle merged expert

把 M5 和 M7 合成一个 expert:

```text
angle_info = cat(angle_ebd, node_for_angle, edge_k, edge_j)
```

输入/输出:

```text
angle_dim -> e_dim + a_dim
```

输出后再 split:

```text
edge_angle1_out, angle_self_out = angle_merged_out.split([e_dim, a_dim])
```

M6 仍是普通本地 MLP:

```text
edge_angle2_out = edge_angle_linear2_moe(edge_angle_agg)
```

它不参与 MoE expert parallel。

## router 如何决定专家

三个 router:

```python
node_router_out = self.node_router(type_embedding)
edge_router_out = self.edge_router(type_embedding)
angle_router_out = self.angle_router(type_embedding)
```

位置: `deepmd/pt/model/network/moe_router.py`

router 输入是 node-level type embedding:

```text
type_embedding: [nb, nloc, n_dim]
```

router 输出:

```text
weights: [N_node, topk]
indices: [N_node, topk]
```

这里 `indices` 是 global expert id, 范围:

```text
0 <= global_eid < n_routing_experts
```

edge 和 angle 的 routing 从 owner node 继承:

```python
edge_weights = edge_weights_node[n2e_index]
edge_indices = edge_indices_node[n2e_index]

angle_weights = angle_weights_node[n2a_index]
angle_indices = angle_indices_node[n2a_index]
```

也就是说:

- node token 按自己的 type embedding routing。
- edge token 使用所属中心原子的 routing。
- angle token 也使用所属中心原子的 routing。

flat path 下, `nb=1`, `nloc=total_atoms`, 所以 router 看到的是:

```text
type_embedding: [1, total_atoms, n_dim]
```

真实 frame 边界不会影响 router, 因为 token 图结构已经在 edge_index/angle_index 里隔离好了。

## MoE 专家参数如何切分

位置: `deepmd/pt/model/network/moe_expert.py`

每类 expert collection 都有 routing experts 和 shared experts。

routing expert 的权重以 3D tensor 存储:

```text
routing_matrix: [num_in, num_out, experts_per_gpu]
routing_bias:   [num_out, experts_per_gpu]
```

每张卡只保存本地 expert slice:

```text
experts_per_gpu = n_routing_experts // ep_size
```

如果:

```text
n_routing_experts = 64
ep_size = 8
```

则每张卡保存 8 个 routing experts:

```text
rank0: expert 0..7
rank1: expert 8..15
...
rank7: expert 56..63
```

global expert 到 owner GPU 的映射:

```text
target_gpu = global_eid // experts_per_gpu
local_eid  = global_eid % experts_per_gpu
```

shared experts 不切分, 每张卡都有完整副本, 最后直接加到 routing expert 输出上。

## MoEDispatchCombine 的多卡流程

位置: `deepmd/pt/model/network/moe_layer.py::MoEDispatchCombine._forward_multi_gpu()`

这是当前专家并行的核心。

### 1. node/edge/angle topk expand + sort

每个 token 会根据 topk router 输出复制成 `topk` 份:

```text
features:     [N, dim]
topk_indices: [N, topk]
topk_weights: [N, topk]

expanded features: [N * topk, dim]
expanded expert:   [N * topk]
expanded weights:  [N * topk]
```

然后按 global expert id 排序。由于:

```text
global_eid = target_gpu * experts_per_gpu + local_eid
```

按 global expert id 排序同时满足:

- 先按目标 GPU 分块。
- 每个目标 GPU 内再按 local expert id 排序。

这样接收端可以用结构化 offset 构建 gather index, 不需要再做一次 expensive argsort。

CUDA 上会优先用 fused kernel:

```python
fused_topk_expand_sort
```

CPU 或不可用时回退到 PyTorch 实现 `_topk_expand_sort()`。

### 2. pack_for_dispatch

node、edge、angle 三类 token 的输入维度不同。为了用一次 all-to-all 传输, 需要打包成统一行宽。

当前 packer 把三类 token 打到统一 packed tensor:

```text
packed: [packed_rows, D_packed_in]
```

CUDA 上优先用:

```python
fused_pack_for_dispatch
```

否则走:

```python
self.packer.pack_for_dispatch(...)
```

同时得到:

```text
send_splits[g] = 发给第 g 个 EP rank 的 packed row 数
```

### 3. exchange metadata

由于每个 rank 发给每个 rank 的 node/edge/angle 数量都不一样, 需要先交换元信息:

```text
send_info[g] = (node_count_to_g, edge_count_to_g, angle_count_to_g)
recv_info = exchange_metadata(send_info, ep_group)
```

接收端据此知道:

```text
recv_node_counts
recv_edge_counts
recv_angle_counts
recv_splits
```

### 4. Dispatch all-to-all

真正发送 feature tensor:

```python
recv_tensor = all_to_all_differentiable(
    packed,
    send_splits,
    recv_splits,
    ep_group,
)
```

这是 differentiable all-to-all。反向传播时梯度也能通过通信回到发送端 token。

### 5. 交换 expert id

feature tensor 发到目标 GPU 后, 接收端还要知道每个 token 应该由哪个本地 expert 算。所以 expert id 也要 all-to-all。

当前优化版把 node/edge/angle expert id 合成一次 int all-to-all:

```python
_exchange_expert_ids_batched(...)
```

这样比三次单独 all-to-all 少两次 NCCL 调用。

### 6. 本地 expert compute

接收端 unpack feature:

```text
node_recv, edge_recv, angle_recv = packer.unpack_from_dispatch(...)
```

node token 里又拆成 M1 和 M2:

```text
node_m1_recv = node_recv[:, :n_dim]
node_m2_recv = node_recv[:, n_dim:]
```

然后:

```python
local_eids = expert_ids % experts_per_gpu
```

根据 local expert id 构建 expert-contiguous 的 gather index:

```text
recv order -> expert-major order
```

再调用 expert collection 的 batched forward:

```python
node_self_experts.forward_expert_batched(...)
node_sym_experts.forward_expert_batched(...)
edge_experts.forward_expert_batched(...)
angle_experts.forward_expert_batched(...)
```

计算完成后再 ungather 回 dispatch 接收顺序。

### 7. pack_for_combine + reverse all-to-all

本地 expert 输出要发回原始 token 所在 rank。

先 pack:

```python
packed_out = self.packer.pack_for_combine(...)
```

再反向 all-to-all:

```python
returned = all_to_all_differentiable(
    packed_out,
    recv_splits,
    send_splits,
    ep_group,
)
```

注意这里 split 方向和 dispatch 相反。

### 8. unpack, unsort, weighted sum

返回后:

```text
node_ret, edge_ret, angle_ret = unpack_from_combine(...)
```

再用 `unsort_idx` 恢复 topk expand 前的 token 顺序:

```text
node_m1_ret = node_m1_ret[node_unsort_idx]
edge_ret    = edge_ret[edge_unsort_idx]
angle_ret   = angle_ret[angle_unsort_idx]
```

最后对 topk 专家输出加权求和:

```text
out[token] = sum_k weight[token,k] * expert_out[token,k]
```

即:

```python
_weighted_sum_topk(...)
```

### 9. 加 shared experts

routing expert 输出完成后, shared expert 输出直接加上:

```python
node_m1_out += node_self_experts.forward_shared(node_m1_input)
node_m2_out += node_sym_experts.forward_shared(node_m2_input)
edge_out    += edge_experts.forward_shared(edge_input)
angle_out   += angle_experts.forward_shared(angle_input)
```

当前代码还尝试用单独 CUDA stream overlap shared expert computation 和 all-to-all/expert id exchange。

## MoE 输出如何回到 RepFlowLayer

`MoEDispatchCombine` 返回:

```text
node_m1_out
node_m2_out
edge_merged_out
angle_merged_out
```

`forward_moe()` 后处理:

1. split merged edge/angle 输出。
2. edge 的 node update 按 `n2e_index` 聚合回 node。
3. angle 的 edge update 按 `eij2a_index` 聚合回 edge。
4. M6 本地 MLP 继续处理 edge angle 聚合结果。
5. `list_update()` 更新 node/edge/angle embedding。

关键聚合:

```text
node_edge_agg = aggregate(node_edge_out * sw, n2e_index)
edge_angle_agg = aggregate(edge_angle1_out * a_sw, eij2a_index)
```

flat path 的 `n2e_index` 和 `n2a_index` 来自 `edge_index/angle_index`, 它们使用的是 flattened local atom index。所以聚合后仍回到 `[1, total_atoms, dim]` 的 synthetic frame 结构。

## flat RepFlows 输出

所有 RepFlowLayer 跑完后:

```python
h2g2 = RepFlowLayer._cal_hg_dynamic(
    edge_ebd_batched,
    h2_batched,
    sw_batched,
    owner=edge_index[0],
    num_owner=nloc,
    nb=1,
    nloc=nloc,
)
```

然后 squeeze 掉 synthetic batch:

```text
node_ebd: [total_atoms, n_dim]
edge_ebd: [n_edge, e_dim]
h2:       [n_edge, 3]
rot_mat:  [total_atoms, dim_emb, 3]
sw:       [n_edge]
```

这些输出再交给 DPA3 和 fitting net。

## fitting_net.forward_flat

位置: `deepmd/pt/model/task/invar_fitting.py::forward_flat()`

当前 fitting net 没有完全原生 flat 化, 而是采用:

```text
flat descriptor
  -> scatter 到 dense [nframes, max_nloc, ...]
  -> 调原来的 fitting_net.forward()
  -> gather valid atom 回 flat
```

核心索引:

```python
atom_counts = ptr[1:] - ptr[:-1]
local_index = flat_index - ptr[batch]

descriptor_batch[batch, local_index] = descriptor
atype_batch[batch, local_index] = atype
```

输出 gather:

```python
valid_atom_mask = arange(max_nloc)[None, :] < atom_counts[:, None]
result_flat[key] = value[valid_atom_mask]
```

所以 descriptor/neighbor/MoE 是 flat 的, fitting 部分暂时 dense 化复用已有代码。

## force 和 virial 的 flat 计算

位置: `deepmd/pt/model/model/make_model.py::_compute_derivatives_flat()`

force:

```python
energy_derv_r = torch.autograd.grad(
    outputs=energy_atomic.sum(),
    inputs=coord,
    create_graph=True,
    retain_graph=True,
)[0]
force = -energy_derv_r
```

输出是:

```text
force: [total_atoms, 3]
```

正好对齐 LMDB mixed collate 后的 flat label force。

virial 当前不是用 `dE/dbox` 直接作为最终 virial, 而是走和 dense 定义一致的形式:

```python
energy_derv_ext = grad(energy_atomic.sum(), extended_coord)
extended_force = -energy_derv_ext
extended_virial = einsum("ik,ij->ikj", extended_force, extended_coord)
energy_derv_c_redu.index_add_(0, extended_batch, extended_virial)
```

即:

```text
virial[frame] = sum_extended_atoms F_ext outer R_ext
```

这里用 `extended_batch` 把 extended atom 的 virial contribution 聚合回真实 frame。

限制:

```text
do_atomic_virial=True 在 flat mixed-batch forward 下尚未实现
```

## loss 如何适配 mixed batch

位置: `deepmd/pt/loss/ener.py::EnergyStdLoss.forward()`

判断:

```python
is_mixed_batch = "ptr" in input_dict and input_dict["ptr"] is not None
```

mixed batch 下:

```python
natoms_per_frame = ptr[1:] - ptr[:-1]
atom_norms = 1.0 / natoms_per_frame
```

energy 和 virial 的 frame-wise loss 使用每帧自己的 atom 数归一化, 而不是使用一个全局 `natoms`。

force loss 则直接:

```python
diff_f = (force_label - force_pred).reshape(-1)
```

因为 force label 和 force pred 都是 `[total_atoms, 3]`。

需要注意:

- generalized force loss 仍有按固定 `natoms` reshape 的假设, mixed-nloc 下需要单独确认或改造。
- `auto_prob_style/block weighting` 与 `mixed_batch=True` 当前不兼容。

## 训练时梯度同步

位置: `deepmd/pt/train/training.py`

标准 DDP 会在 backward 时把所有参数梯度按 world all-reduce, 但 MoE EP 下这样不对:

- routing expert 参数只存在于某些 EP rank 上。
- shared/non-routing 参数在所有 rank 上都有副本。

因此 `use_moe_ep=True` 时训练循环使用:

```python
with self.wrapper.no_sync():
    loss.backward()
sync_moe_gradients(...)
```

`sync_moe_gradients()` 的规则:

### DP=1, EP=world

这就是 8 卡 `moe_ep_size=8` 的情况。

- routing expert 梯度通过 differentiable all-to-all backward 已经回到本地 expert。
- 非 routing 参数需要 world all-reduce 后除以 `world_size`。

### DP>1, EP<world

- routing expert 参数在同一 DP column 中有副本, 所以 routing expert grad 在 `dp_group` 里 all-reduce。
- 非 routing 参数仍在 world 里 all-reduce。
- routing expert grad 除以 `world_size`, 因为 EP 内 all-to-all backward 已经聚合了来自 EP ranks 的 token 梯度。

routing expert 参数通过名字识别:

```text
.routing_matrix
.routing_bias
.routing_experts.  # legacy
```

## checkpoint 保存和加载

位置: `deepmd/pt/utils/moe_checkpoint.py`

EP 训练时, 每张卡只保存一部分 routing experts。如果直接保存本地 state_dict, rank0 只会写出 rank0 的专家切片。因此 `Trainer.save_model()` 在 `use_moe_ep=True` 时调用:

```python
moe_state_dict_to_global(...)
```

它对所有 routing 3D tensor 做:

```text
rank local routing_matrix: [I, O, experts_per_gpu]
all_gather over ep_group
concat last dim
global routing_matrix: [I, O, n_routing_experts]
```

`routing_bias` 同理:

```text
[O, experts_per_gpu] -> [O, n_routing_experts]
```

非 routing 参数直接取本地副本。

加载时 `moe_load_state_dict_from_global()` 反过来切片:

```python
start = ep_rank * experts_per_gpu
end = start + experts_per_gpu
local_tensor = global_tensor[..., start:end]
```

这允许保存 checkpoint 时是 global expert layout, 加载时再按当前 EP size reshard。

## 当前 flat MoE 的完整调用逻辑

下面是把数据、flat graph、MoE EP、loss 串起来的完整调用链:

```text
torchrun ... dp --pt train input.json
  -> deepmd/pt/entrypoints/main.py::get_trainer
       -> detect dpa3 repflow.use_moe and training.moe_ep_size
       -> init_ep_dp_groups()
       -> LmdbDataset(... mixed_batch=True)

  -> Trainer.__init__
       -> set_moe_ep_context(ep_group, ep_rank, ep_size)
       -> get_model_for_wrapper()
            -> DPA3 reads EP context
            -> RepFlows builds RepFlowLayer(use_moe=True)
            -> RepFlowLayer builds routers and MoEDispatchCombine
       -> get_data_loader()
            -> DataLoader(... collate_fn=_collate_lmdb_mixed_batch)

  -> training loop
       -> Trainer.get_data()
            -> batch_data = next(iterator)
            -> atom-wise tensors already flat
            -> move tensors to DEVICE
            -> build_precomputed_flat_graph() on DEVICE
            -> input_dict includes coord/atype/box + flat graph fields
            -> label_dict includes energy/force/virial labels

       -> ModelWrapper.forward(...)
            -> batch/ptr detected
            -> model(**input_dict)

       -> EnergyModel.forward(...)
            -> forward_common_flat()
            -> forward_common_flat_native()
                 -> rebuild_extended_coord_from_flat_graph()
                 -> forward_common_lower_flat()

       -> DPAtomicModel.forward_common_atomic_flat()
            -> DPA3.forward_flat()
                 -> RepFlows.forward_flat()
                      -> prod_env_mat_flat(nlist_ext)
                      -> dynamic edge/angle flatten
                      -> for each RepFlowLayer:
                           -> RepFlowLayer.forward(... type_embedding=...)
                           -> forward_moe()
                                -> routers produce topk experts
                                -> MoEDispatchCombine()
                                     -> topk expand/sort
                                     -> pack node/edge/angle
                                     -> metadata exchange
                                     -> dispatch all-to-all
                                     -> expert id all-to-all
                                     -> local expert compute
                                     -> pack outputs
                                     -> combine all-to-all
                                     -> unsort + weighted topk sum
                                     -> add shared experts
                                -> aggregate edge/angle updates
                                -> list_update node/edge/angle
                      -> rot_mat/h2g2
                 -> descriptor output
            -> fitting_net.forward_flat()
                 -> scatter flat descriptor to dense [nframes,max_nloc]
                 -> ordinary fitting forward
                 -> gather valid atoms back to flat
            -> atomic energy output

       -> forward_common_lower_flat()
            -> index_add atom energy by batch -> frame energy

       -> _compute_derivatives_flat()
            -> force from grad wrt coord
            -> virial from extended_force outer extended_coord, index_add by extended_batch

       -> EnergyStdLoss.forward()
            -> mixed-batch normalization by ptr
            -> force loss on flat [total_atoms,3]

       -> backward
            -> if use_moe_ep:
                 wrapper.no_sync()
                 loss.backward()
                 sync_moe_gradients()
            -> optimizer.step()
```

## flat path 和普通 dense path 的等价性关键

为了让 same-nloc 数据在两条路径下结果一致, 当前实现依赖以下不变量:

1. same-nloc flat graph 分支使用 dense-like 批量构图, 再 flatten。
2. flat graph 在 `DEVICE` 上构建, 和 dense 路径使用同一设备上的 neighbor list 行为。
3. neighbor list 对等距候选有 deterministic tie-break, 避免 CPU/GPU 或排序不稳定导致邻居顺序不同。
4. `mapping` 把 ghost neighbor 映射回真实 atom embedding。
5. `nlist_ext/a_nlist_ext` 用于取 extended coordinate 算距离。
6. `nlist/a_nlist` 用于 node embedding 和 edge/angle graph 索引。
7. flat path 必须把 `type_embedding` 传给 RepFlowLayer, 否则 MoE router 不是同一逻辑。
8. virial 用 extended force 与 extended coordinate 聚合, 与 dense 定义保持一致。

## 与普通 MoE EP 文档的关系

已有文档 `history/2026-05-20_MOE_EXPERT_PARALLELISM_UNDERSTANDING.md` 主要讲普通 MoE EP:

- EP/DP group
- router
- expert collection
- MoEDispatchCombine
- checkpoint

本文档补充的是 flat mixed-batch 如何接入这套 MoE EP:

- LMDB mixed batch flatten
- flat graph 构建
- `forward_flat` 调用链
- flat token 如何进入 router 和 expert parallel
- flat force/virial/loss 如何收尾

一句话区分:

```text
普通 MoE EP 文档解释专家并行怎么做。
本文档解释 flat mixed-batch 数据怎么走到同一套专家并行里。
```

## 当前限制和风险点

1. `mixed_batch=True` 当前只支持 DPA3/RepFlow 这类 flat-graph capable descriptor。
2. flat graph 字段缺失会直接报错, 模型内部不会自动兜底构图。
3. `auto_prob_style/block weighting` 与 LMDB `mixed_batch=True` 还不兼容。
4. `do_atomic_virial=True` 在 flat mixed-batch forward 下尚未实现。
5. generalized force loss 对 mixed-nloc 仍需额外确认。
6. MoE 依赖 dynamic selection 和固定维度比例, 不是所有 RepFlow 配置都能打开。
7. `n_routing_experts` 必须能被 `moe_ep_size` 整除。
8. 8 卡 `moe_ep_size=8` 时 `dp_size=1`, routing expert 没有 DP 副本, 非 routing 参数仍需要 world grad sync。

## 已验证的等价性场景

当前仓库中已经构造了 same-nloc 的 pkl 对比测试:

```text
test_mptraj/same_nloc_ep_compare.pkl
test_mptraj/compare_same_nloc_pkl_ep.py
test_mptraj/same_nloc_ep_compare_result.json
```

测试目标:

```text
同一批 same-nloc 数据
  -> dense/same-nloc 数据处理方式
  -> flat mixed-batch 数据处理方式
  -> 都用 8 卡专家并行
  -> 对比 energy / atom_energy / force / virial
```

已通过的结果:

```text
energy max_abs:      1.7881393432617188e-07
atom_energy max_abs: 1.4901161193847656e-07
force max_abs:       6.257351969907177e-09
virial max_abs:      6.507677685618773e-08
passed:              true
```

这说明在 same-nloc 条件下, 当前 flat path 和原处理路径在 MoE EP 下可以达到数值等价。

## 需要记住的核心点

flat MoE 并行的核心不是“给 mixed batch 写一套新专家并行”, 而是:

1. 用 `batch/ptr` 描述真实 frame 边界。
2. 用 `extended_* / mapping / *_ext` 描述 PBC ghost 和邻居关系。
3. 用 `edge_index/angle_index` 把 DPA3 dynamic graph 变成 flat token 图。
4. 在 RepFlows 里把 flat atoms 包成 `[1, total_atoms, dim]`。
5. 让 RepFlowLayer 的 MoE router 和 MoEDispatchCombine 继续处理 node/edge/angle token。
6. 在输出阶段用 `batch` 和 `extended_batch` 把 atom/extended atom 结果聚合回 frame。

所以真正的边界是:

```text
flat graph 负责数据形态和邻居拓扑
MoE EP 负责 token 到专家的跨 GPU dispatch/compute/combine
```
