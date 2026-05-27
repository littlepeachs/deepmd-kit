# GP + LMDB mixed batch + MoE 实现记录

日期: 2026-05-24

仓库: `/aisi-nas/liwentao/deepmd-kit-moe`

分支: `0415_moe_pack`

记录对象: 在当前 MoE flat mixed-batch 路径上融合主图并行 GP, 目标是让一个 LMDB mixed batch 被 broadcast 到所有 GP ranks, 每个 rank 只计算本 rank 的子图, 子图内部再走已有 MoE 专家并行。

参考对象:

```text
/aisi-nas/liwentao/deepmd-kit-moe/deepmd-kit-arch
history/2026-05-21_ARCH_GRAPH_PARALLEL_UNDERSTANDING.md
history/2026-05-20_ARCH_LMDB_MIXED_BATCH_FORWARD_UNDERSTANDING.md
history/2026-05-20_FLAT_MOE_EXPERT_PARALLELISM_UNDERSTANDING.md
```

## 一句话总结

这次实现的第一阶段不是引入通用的 DP x GP x EP mesh, 而是把 GP 维度复用到当前 EP rank 集合上:

```text
world_size == moe_ep_size == graph_parallel_size
dp_size == 1
每个 rank 拿同一个 LMDB mixed batch
每个 rank 拥有同一个模型公共部分 + 本 rank 的 MoE routing expert shard
每个 rank 只 forward 该 batch 的一个 flat atom 子图
MoE dispatch 只 dispatch 本 rank local node/edge/angle token
能量/力/virial/梯度在 GP group 上合并
```

直观调用链:

```text
torchrun 8 ranks
  -> dp --pt train input.json
  -> main.get_trainer()
  -> init_ep_dp_groups(moe_ep_size)
  -> dist.new_group(same ranks) as gp_group
  -> Trainer.get_data()
  -> rank0 next(LMDB DataLoader), broadcast to all ranks
  -> build_precomputed_flat_graph()
  -> build_flat_graph_partition()
  -> ModelWrapper.forward(flat_graph_partition=...)
  -> EnergyModel / DPAtomicModel / DPA3
  -> RepFlows.forward_flat_gp()
  -> RepFlowLayer.forward_moe_gp()
  -> local fitting
  -> gather atom outputs, reduce frame outputs
  -> local-energy grad wrt global coord, GP all-reduce force/virial
  -> manual grad sync with GP sum semantics
```

## 使用方式

当前验证使用的配置是:

```text
/aisi-nas/liwentao/deepmd-kit-moe/test_mptraj/input.json
```

关键字段:

```json
{
  "training": {
    "training_data": {
      "systems": "/aisi-nas/liwentao/mptraj_v024.lmdb",
      "batch_size": 1,
      "mixed_batch": true
    },
    "validation_data": {
      "systems": "/aisi-nas/liwentao/mptraj_v024.lmdb",
      "batch_size": 1,
      "mixed_batch": true,
      "numb_btch": 1
    },
    "numb_steps": 5000,
    "moe_ep_size": 8,
    "graph_parallel": true,
    "graph_parallel_size": 8
  },
  "model": {
    "descriptor": {
      "type": "dpa3",
      "repflow": {
        "use_dynamic_sel": true,
        "smooth_edge_update": true,
        "use_moe": true,
        "n_routing_experts": 64,
        "moe_topk": 8
      }
    }
  }
}
```

运行命令:

```bash
cd /aisi-nas/liwentao/deepmd-kit-moe/test_mptraj
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/aisi-nas/liwentao/miniconda/dpa3_moe/bin/torchrun --standalone --nproc_per_node=8 \
  /aisi-nas/liwentao/miniconda/dpa3_moe/bin/dp --pt train --skip-neighbor-stat input.json
```

当前第一阶段强约束:

```text
graph_parallel = true
descriptor.type = dpa3
repflow.use_dynamic_sel = true
repflow.smooth_edge_update = true
fitting_net.type omitted or ener
single-task training
LMDB mixed_batch = true
moe_ep_size == graph_parallel_size == world_size
```

这些约束集中在:

```text
deepmd/pt/train/training.py::validate_graph_parallel_config()
deepmd/pt/entrypoints/main.py::get_trainer()
```

## 入口建组逻辑

入口在:

```text
deepmd/pt/entrypoints/main.py::get_trainer()
```

实现逻辑:

```text
读取 config
  -> validate_graph_parallel_config(config)
  -> 如果 graph_parallel=true, 要求 torch.distributed 已初始化
  -> 检查 descriptor.type=dpa3
  -> 检查 moe_ep_size 和 graph_parallel_size
  -> init_ep_dp_groups(moe_ep_size)
  -> 复用 EP ranks 创建单独 gp_group
  -> set_graph_parallel_context(True, gp_group, ep_rank, ep_size, reduce_backward=False)
  -> Trainer(... use_moe_ep=..., use_graph_parallel=True, gp_group=...)
```

为什么 GP group 和 EP group 使用同一批 ranks, 但仍然单独创建 group:

1. 当前需求里 GP 和 EP 是同一个并行维度, 都覆盖全部 8 张卡。
2. 单独的 GP context 可以保证原始 MoE EP 路径不被改坏。
3. 后续如果要从 `gp_size == ep_size` 扩展到真正 `DP x GP x EP`, 可以把 GP group 替换掉, 而不需要改 MoE EP 的上下文。

`reduce_backward=False` 是一个关键点。GP output 的 frame energy / atom energy gather 是 forward 合并语义, backward 阶段不能再隐式 all-reduce 一次, 否则 shared parameter 梯度会被重复计数。真正的梯度同步放在训练循环里手动做。

## LMDB mixed batch broadcast

数据读取入口:

```text
deepmd/pt/train/training.py::Trainer.get_data()
```

原始 flat mixed-batch 是每个 rank 各自从 DataLoader 取一个 batch。GP 下不允许这样做, 因为 GP group 内所有 ranks 必须切同一个图。

当前实现:

```text
if use_graph_parallel:
  rank0: batch_data = next(iterator)
  other ranks: batch_data = None
  batch_data = broadcast_batch_data(batch_data, group=gp_group)
else:
  batch_data = next(iterator)
```

通信工具:

```text
deepmd/pt/utils/graph_parallel.py::broadcast_batch_data()
```

实现上使用 `torch.distributed.broadcast_object_list`, 适合第一阶段直接传一个 batch dict。第一次 batch 会额外校验 `ptr/batch/coord` 一致性, 避免某些 rank 误拿不同图后继续训练。

这样做的语义是:

```text
GP group 内所有 ranks 有相同:
  coord
  atype
  box
  ptr
  batch
  labels
  precomputed flat graph
```

区别只在:

```text
每个 rank 的 flat_graph_partition 不同
```

## flat graph 构建和子图切分

flat graph 仍然沿用当前 LMDB mixed-batch 的构图逻辑:

```text
Trainer.get_data()
  -> batch_data tensors move to DEVICE
  -> build_precomputed_flat_graph()
```

构图位置:

```text
deepmd/pt/utils/nlist.py::build_precomputed_flat_graph()
```

它输出 flat path 需要的字段:

```text
nlist
nlist_ext
a_nlist
a_nlist_ext
nlist_mask
a_nlist_mask
edge_index
angle_index
extended_coord
extended_atype
extended_batch
ptr
batch
...
```

GP 子图切分位置:

```text
deepmd/pt/utils/graph_parallel_flat.py::build_flat_graph_partition()
```

切分原则:

```text
total_atoms = int(ptr[-1])
按 flat atom index 做连续均衡切分
rank r 拥有 [local_start, local_end)
edge 按中心 atom edge_index[0] 归属
angle 按中心 atom angle_index[0] 归属
angle 中引用的 edge id 会 remap 成 local edge id
```

返回对象:

```text
FlatGraphPartition(
  rank,
  world_size,
  local_start,
  local_end,
  local_size,
  atom_index,
  batch,
  edge_mask,
  edge_ids,
  edge_index,
  angle_mask,
  angle_ids,
  angle_index,
)
```

这里 edge 和 angle 都按 owner/center atom 切分, 但 neighbor atom 可以在别的 rank。为了解决跨 rank neighbor 读取, RepFlow 每一层输入仍保留 full node embedding, local edge 只从 full node embedding 里 `index_select` neighbor。

## partition 如何传入模型

模型 plumbing 改动的目的只有一个: 当 `flat_graph_partition is None` 时保持原始 flat/MoE 路径不变; 当它非空时走 GP 路径。

传递链路:

```text
Trainer.get_data()
  -> input_dict["flat_graph_partition"] = partition

ModelWrapper.forward()
  -> model(... flat_graph_partition=...)

EnergyModel.forward()
  -> atomic_model(... flat_graph_partition=...)

DPAtomicModel.forward_common_atomic_flat()
  -> descriptor.forward_flat(... flat_graph_partition=...)
  -> fitting_net.forward_flat(... atom_index=partition.atom_index, ...)

DPA3.forward_flat()
  -> repflows.forward_flat(... flat_graph_partition=...)

RepFlows.forward_flat()
  -> if partition is not None: forward_flat_gp()
  -> else: 原始 forward_flat()
```

涉及文件:

```text
deepmd/pt/train/wrapper.py
deepmd/pt/model/model/ener_model.py
deepmd/pt/model/atomic_model/dp_atomic_model.py
deepmd/pt/model/descriptor/dpa3.py
deepmd/pt/model/descriptor/repflows.py
deepmd/pt/model/task/invar_fitting.py
deepmd/pt/model/model/make_model.py
```

一个容易漏的点是 `concat_output_tebd=True`。DPA3 原始逻辑在 descriptor 输出后会拼 type embedding。GP 下 descriptor 是 local atoms, 所以 type embedding 也必须切成 `[local_start:local_end]` 再 concat, 不能拼全图 type embedding。

## RepFlows.forward_flat_gp 的核心逻辑

实现位置:

```text
deepmd/pt/model/descriptor/repflows.py::RepFlows.forward_flat_gp()
```

它和原始 `forward_flat()` 最大不同是: 原始 flat 路径每层都处理全图 node/edge/angle; GP 路径每个 rank 只处理本 rank 的 local edge/angle 和 local center node, 但读取 neighbor 时仍使用 full node embedding。

伪代码:

```text
node_ebd_batched: [1, total_atoms, n_dim]  # full
local range: [local_start, local_end)

构 local env:
  local_nlist_ext = nlist_ext[local_start:local_end]
  local_nlist_mask = nlist_mask[local_start:local_end]
  local_a_nlist_ext = a_nlist_ext[local_start:local_end]
  local_a_nlist_mask = a_nlist_mask[local_start:local_end]

构 local edge embedding:
  edge_input = dmatrix[local_nlist_mask]
  edge_ebd = edge_embd(edge_input)

构 local angle embedding:
  先从 local a_nlist 构 angle_input
  再用 partition.angle_mask 对齐到 local_angle_index
  angle_ebd = angle_embd(angle_input)

for layer in layers:
  if use_moe:
    local_type_embedding = atype_embd[:, local_start:local_end, :]
    local_node, local_edge, local_angle = layer.forward_moe_gp(
      full_node_ebd,
      local_edge_ebd,
      local_angle_ebd,
      local_edge_index,
      local_angle_index,
      local_type_embedding,
      local_start,
    )
  else:
    local_node, local_edge, local_angle = layer.forward_gp(...)

  if not last layer:
    full_node_ebd = gather_node_tensor(local_node)

return local descriptor, local rot_mat
```

为什么每层结束后要 gather full node:

```text
下一层的 local edge 仍可能连接到其他 rank 的 neighbor atom。
如果只保留 local node, 下一层无法读跨 rank neighbor embedding。
```

为什么最后一层不 gather:

```text
fitting 只需要本 rank local atom descriptor。
energy/force/virial 会在上层按 GP group 合并。
```

## GP 版本的 MoE layer

实现位置:

```text
deepmd/pt/model/descriptor/repflow_layer.py::RepFlowLayer.forward_moe_gp()
```

输入语义:

```text
node_ebd_ext: [1, total_atoms, n_dim]   # full node embedding
edge_ebd: [n_local_edge, e_dim]
angle_ebd: [n_local_angle, a_dim]
edge_index: [2, n_local_edge]
angle_index: [3, n_local_angle]
type_embedding: [1, n_local_node, n_dim]
local_start: int
```

index 转换:

```text
n2e_index = edge_index[0] - local_start       # local owner node
n_ext2e_index = edge_index[1]                 # global neighbor node
n2a_index = angle_index[0] - local_start      # local owner node
eij2a_index = angle_index[1]                  # local edge id
eik2a_index = angle_index[2]                  # local edge id
```

关键设计:

1. Router 只看 local token:

```text
node_router_out = node_router(type_embedding)
edge_router_out = edge_router(type_embedding)
angle_router_out = angle_router(type_embedding)
```

2. MoE dispatch 仍使用已有 expert-parallel all-to-all:

```text
moe_phase1(...)
```

也就是说, GP 只减少本 rank 要 dispatch 的 token 数; dispatch 到哪个 GPU 上的 expert 仍由原始 MoE EP 机制决定。

3. neighbor embedding 从 full node 读取:

```text
nei_node_ebd = index_select(node_ebd_ext.reshape(-1, n_dim), n_ext2e_index)
```

4. 输出只返回 local node/edge/angle:

```text
node_ebd: [1, n_local_node, n_dim]
edge_ebd: [n_local_edge, e_dim]
angle_ebd: [n_local_angle, a_dim]
```

## non-MoE GP 路径

为了验证 GP 本身, 同时实现了非 MoE 版本:

```text
deepmd/pt/model/descriptor/repflow_layer.py::RepFlowLayer.forward_gp()
```

它和 `forward_moe_gp()` 的区别是:

```text
不走 router
不走 MoE all-to-all
只在本 rank local edge/angle/node 上做普通 RepFlow update
每层之间仍然 gather full node
```

这个路径用于 same-nloc 测试, 证明切图、local fitting、output reduce、force/virial reduce 本身是正确的。

## local fitting

实现位置:

```text
deepmd/pt/model/task/invar_fitting.py::InvarFitting.forward_flat()
```

原始 flat fitting 输入是全图 descriptor:

```text
descriptor: [total_atoms, dim]
atype: [total_atoms]
batch: [total_atoms]
ptr: [nframes + 1]
```

GP 下输入变成:

```text
descriptor_local: [local_atoms, dim]
atype_local: [local_atoms]
batch_local: [local_atoms]
atom_index: [local_atoms]  # 全局 flat atom index
ptr_global: [nframes + 1]
```

因此 `forward_flat()` 增加了 `atom_index` 参数。它先把 local flat atom 投影回 dense frame-local 位置:

```text
local_index = atom_index - ptr[batch]
descriptor_batch[batch, local_index] = descriptor
atype_batch[batch, local_index] = atype
```

然后复用原始 dense fitting:

```text
self.forward(descriptor_batch, atype_batch, ...)
```

最后只取回 local atoms 的输出:

```text
result_flat[key] = value[batch, local_index]
```

这样 fitting 网络本身不需要知道 GP, 仍复用原始 per-frame dense fitting 逻辑。

## 能量、原子能量、mask 的合并

实现位置:

```text
deepmd/pt/model/model/make_model.py
```

local fitting 之后每个 rank 只拿到:

```text
energy_atomic_local: [local_atoms, 1]
```

处理逻辑:

```text
model_ret["energy_local"] = energy_atomic_local
model_ret["energy"] = gather_node_tensor(energy_atomic_local, backward=False)

energy_redu_local = zeros([nframes, 1])
energy_redu_local.index_add_(0, local_batch, energy_atomic_local)
model_ret["energy_redu"] = reduce_graph_tensor(energy_redu_local)
```

语义:

```text
atom_energy:
  gather 回全局 flat atom 顺序, 用于输出和 loss 对齐

energy:
  每个 rank 先算自己的 frame partial sum
  再在 GP group 上 sum
```

`mask` 如果存在也按 atom 维 gather 回全局顺序。

## force 和 virial

force 使用 local energy 对 global coord 求导:

```text
energy_sum = energy_local.sum()
energy_derv_r = autograd.grad(energy_sum, coord, create_graph=True)
force_partial = -energy_derv_r
force = reduce_graph_tensor(force_partial)
```

原因:

```text
每个 rank 的 local energy 可能依赖全局 coord 中任意 neighbor atom。
所以每个 rank 都对完整 coord 求一份 partial force。
最后 GP all-reduce 得到完整 force。
```

virial 当前保持 DeePMD flat 原有语义: 用 extended coordinate 上的梯度构 frame virial:

```text
energy_derv_ext = autograd.grad(energy_local.sum(), extended_coord, create_graph=True)
extended_force = -energy_derv_ext
extended_virial = einsum(extended_force, extended_coord)
frame_virial_local.index_add_(0, extended_batch, extended_virial)
frame_virial = reduce_graph_tensor(frame_virial_local)
```

这里没有改成简单 `force * coord`, 因为原始 flat/Dense 对齐路径就是基于 extended coordinate 的 virial 定义。same-nloc 测试中 GP 和 full flat 的 virial 已经数值对齐。

## GP collectives 和 higher-order autograd

通信工具集中在:

```text
deepmd/pt/utils/graph_parallel.py
```

主要接口:

```text
set_graph_parallel_context()
clear_graph_parallel_context()
graph_parallel_enabled()
gather_node_tensor()
reduce_graph_tensor()
broadcast_batch_data()
assert_batch_consistent()
```

`gather_node_tensor()` 用于两类场景:

1. 层间 node embedding gather:

```text
local node -> all_gather -> full node
```

这类 gather 必须支持 autograd, 因为下一层 local computation 的梯度要回到上一层 local node。

2. 最终 atom output gather:

```text
local atom_energy -> global atom_energy
```

这类 gather 只用于输出/loss 对齐, backward 不应该再次触发 GP reduce。因此 GP context 在训练入口设置 `reduce_backward=False`, 最终输出 gather 用 forward-only backward 语义。

`reduce_graph_tensor()` 用于 frame energy、force、virial 等需要跨 GP rank 求和的张量。

MoE all-to-all 和 GP all-gather/all-reduce 在 force/virial 的二阶 autograd 中会交织。为避免不同 rank 以不同顺序进入 collective, 增加了:

```text
deepmd/pt/utils/collective_order.py
```

训练循环在 `use_moe_ep and use_graph_parallel` 时使用:

```text
with collective_ordering():
  forward
  backward
```

它通过零值 tensor dependency 把 collective autograd Function 串成同一顺序, 不改变数值。

同时 GP backward 期间关闭 autograd 多线程:

```text
with torch.autograd.set_multithreading_enabled(False):
  loss.backward()
```

这是为了让同一 rank 内多个二阶 collective 分支也按确定顺序遍历。

## 梯度同步逻辑

实现位置:

```text
deepmd/pt/train/training.py
deepmd/pt/utils/moe_ep_dp.py::sync_moe_gradients()
```

普通 DDP 的语义是:

```text
每个 rank 处理不同 batch replica
backward 时 world all-reduce 并平均梯度
```

GP 语义不同:

```text
每个 rank 处理同一个 batch 的不同图 shard
shared params 的梯度应该 sum, 不是 average
```

因此训练循环在 MoE 或 GP 下禁用 DDP 自动同步:

```text
with self.wrapper.no_sync():
  loss.backward()
sync_moe_gradients(..., non_routing_divisor=1.0 if use_graph_parallel else None)
```

`sync_moe_gradients()` 的规则:

```text
routing expert params:
  在 DP group 内 all-reduce
  当前第一阶段 dp_size=1, 所以不额外跨 DP 同步
  expert token 梯度由 MoE all-to-all backward 回到对应 expert shard

non-routing/shared params:
  world group all-reduce
  EP-only 原始语义下除以 world_size
  GP 语义下除以 1.0, 即求和
```

为了避免某些 rank 某个 expert 参数 `grad=None` 而其他 rank 有梯度导致 collective 不一致, 增加了 `_ensure_grad_for_collective()`:

```text
先 all-reduce 一个 has_grad flag
如果组内有人有 grad, 本 rank grad=None 则补 zeros_like
然后所有 rank 都进入同一个 all-reduce
```

这个修复对未来 `dp_size>1` 的 routing expert 同步也必要。

## 为什么不破坏原有 MoE

隔离策略:

1. `training.graph_parallel` 默认为 false。
2. 所有 GP 入口都由 `flat_graph_partition is not None` 或 `use_graph_parallel` 控制。
3. `forward_flat()` 原路径仍保留, 只有 partition 非空才 dispatch 到 `forward_flat_gp()`。
4. MoE 原有 `forward_moe()` 不改语义, 新增 `forward_moe_gp()` 专门处理 local token。
5. EP group 初始化仍使用原始 `init_ep_dp_groups()`, GP 单独建 context。
6. 梯度同步函数保持默认 `non_routing_divisor=None` 时的原始 EP/DP averaging 语义; GP 显式传 `1.0`。

## 验证记录

### 1. 单进程错误保护

直接用单进程跑 GP 会提前报错:

```text
ValueError: training.graph_parallel=True requires distributed launch with WORLD_SIZE > 1 via torchrun.
```

说明 GP 不会静默退化成错误的单 rank 训练。

### 2. non-MoE same-nloc 数值对齐

命令:

```bash
cd /aisi-nas/liwentao/deepmd-kit-moe/test_mptraj
CUDA_VISIBLE_DEVICES=0,1 \
/aisi-nas/liwentao/miniconda/dpa3_moe/bin/torchrun --standalone --nproc_per_node=2 \
  ./compare_same_nloc_pkl_gp.py \
  --input input.json \
  --pkl same_nloc_ep_compare.pkl \
  --disable-moe \
  --ep-size 2 \
  --check-gradients \
  --result-json same_nloc_gp_compare_nonmoe_grad_result.json
```

结果:

```text
passed = true
energy max_abs = 3.576e-7
atom_energy max_abs = 1.192e-7
force max_abs = 2.129e-9
virial max_abs = 2.213e-8
checked_non_routing_params = 120
gradient max_abs = 1.526e-5
gradient passed = true
```

### 3. MoE same-nloc 数值对齐

命令:

```bash
cd /aisi-nas/liwentao/deepmd-kit-moe/test_mptraj
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/aisi-nas/liwentao/miniconda/dpa3_moe/bin/torchrun --standalone --nproc_per_node=8 \
  ./compare_same_nloc_pkl_gp.py \
  --input input.json \
  --pkl same_nloc_ep_compare.pkl \
  --ep-size 8 \
  --check-gradients \
  --result-json same_nloc_gp_compare_moe_grad_result.json
```

结果:

```text
passed = true
energy max_abs = 3.576e-7
atom_energy max_abs = 2.384e-7
force max_abs = 3.094e-9
virial max_abs = 4.045e-8
checked_non_routing_params = 121
gradient max_abs = 1.526e-5
gradient passed = true
```

### 4. LMDB mixed batch 1 step smoke

8 卡 GP+MoE+LMDB mixed batch 1 step 通过, 能完成:

```text
EP/GP init
Start to train 1 steps
Batch 1
Saved model
exit_status=0
```

### 5. LMDB mixed batch 5000 step

命令:

```bash
cd /aisi-nas/liwentao/deepmd-kit-moe/test_mptraj
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/aisi-nas/liwentao/miniconda/dpa3_moe/bin/torchrun --standalone --nproc_per_node=8 \
  /aisi-nas/liwentao/miniconda/dpa3_moe/bin/dp --pt train --skip-neighbor-stat input.json
```

日志:

```text
/tmp/gp8_moe_lmdb_5000_after_gradfix_20260523_091547.log
```

结果:

```text
exit_status=0
Batch 5000 completed
Saved model to model_lmdb_mixed_5000-5000.pt
Trained model has been saved to: model_lmdb_mixed_5000
average training time: 0.5034 s/batch (100 batches excluded)
```

最终文件:

```text
test_mptraj/model_lmdb_mixed_5000-5000.pt
test_mptraj/model_lmdb_mixed_5000.pt -> model_lmdb_mixed_5000-5000.pt
test_mptraj/lcurve.out  # 已写到 5000
```

### 6. 静态检查

目标文件检查通过:

```bash
/root/miniconda3/bin/ruff check \
  deepmd/pt/entrypoints/main.py \
  deepmd/pt/train/training.py \
  deepmd/pt/train/wrapper.py \
  deepmd/pt/model/model/ener_model.py \
  deepmd/pt/model/atomic_model/dp_atomic_model.py \
  deepmd/pt/model/descriptor/dpa3.py \
  deepmd/pt/model/descriptor/repflows.py \
  deepmd/pt/model/descriptor/repflow_layer.py \
  deepmd/pt/model/task/invar_fitting.py \
  deepmd/pt/model/model/make_model.py \
  deepmd/pt/model/network/moe_ep_ops.py \
  deepmd/pt/model/network/moe_layer.py \
  deepmd/pt/utils/moe_ep_dp.py \
  deepmd/pt/utils/graph_parallel.py \
  deepmd/pt/utils/graph_parallel_flat.py \
  deepmd/pt/utils/collective_order.py \
  test_mptraj/compare_same_nloc_pkl_gp.py
```

```bash
/root/miniconda3/bin/ruff format --check <same target files>
```

```bash
/aisi-nas/liwentao/miniconda/dpa3_moe/bin/python -m py_compile <same target files>
```

全仓库 `ruff check .` 当前不作为本次通过标准, 因为仓库里已有参考目录和历史调试文件存在无关 lint 问题。

## 当前限制

当前实现是可训练的第一阶段版本, 不是最终通用并行拓扑。明确限制如下:

1. 只支持 `gp_size == moe_ep_size == world_size`。
2. 只支持单任务训练。
3. 只支持 DPA3 + RepFlow dynamic selection。
4. 只支持 ordinary energy fitting, 不支持 `direct_force_ener` 等特殊 fitting。
5. 只支持 LMDB `mixed_batch=true`。
6. GP + MoE 下 routing expert 参数的直接数值梯度对齐还没有单独拆出来测; 当前验证覆盖了 MoE forward/backward 路径和 non-routing 梯度。
7. mixed-nloc 的独立数值对齐测试还没补; 真实 LMDB mixed-batch 1 step 和 5000 step 已通过。
8. `dp_size>1` 的 DP x GP x EP 还没有启用, 但 routing expert zero-grad sync 已按未来 general case 修过。
9. checkpoint resume 没有单独做回归。
10. 通信效率还可以继续优化, 目前每层 full node gather, batch broadcast 也是 object broadcast。

## 后续建议

下一阶段如果要走通用 DP x GP x EP, 建议按这个顺序推进:

1. 把 rank 拓扑从 `ep_size == gp_size == world_size` 改成显式 mesh。
2. DataLoader 只在同一个 GP group 内 broadcast, 不同 DP group 拿不同 batch。
3. DDP group 改成同一 GP shard 维度上的 DP replicas。
4. routing expert 梯度在 DP group 内做数值对齐测试。
5. 增加 mixed-nloc same-batch fixture, 对齐 full flat 与 GP outputs/gradients。
6. 增加 checkpoint save/load/resume 的 GP+MoE 回归。
7. 优化层间通信, 尝试只 gather 必要 neighbor node embedding, 而不是每层 full node gather。

## 关键文件索引

新增:

```text
deepmd/pt/utils/graph_parallel.py
deepmd/pt/utils/graph_parallel_flat.py
deepmd/pt/utils/collective_order.py
test_mptraj/compare_same_nloc_pkl_gp.py
```

核心修改:

```text
deepmd/pt/entrypoints/main.py
deepmd/pt/train/training.py
deepmd/pt/train/wrapper.py
deepmd/pt/model/model/ener_model.py
deepmd/pt/model/model/make_model.py
deepmd/pt/model/atomic_model/dp_atomic_model.py
deepmd/pt/model/descriptor/dpa3.py
deepmd/pt/model/descriptor/repflows.py
deepmd/pt/model/descriptor/repflow_layer.py
deepmd/pt/model/task/invar_fitting.py
deepmd/pt/model/network/moe_ep_ops.py
deepmd/pt/model/network/moe_layer.py
deepmd/pt/utils/moe_ep_dp.py
```

配置和验证:

```text
test_mptraj/input.json
test_mptraj/same_nloc_ep_compare.pkl
test_mptraj/same_nloc_gp_compare_nonmoe_grad_result.json
test_mptraj/same_nloc_gp_compare_moe_grad_result.json
test_mptraj/lcurve.out
test_mptraj/model_lmdb_mixed_5000-5000.pt
```
