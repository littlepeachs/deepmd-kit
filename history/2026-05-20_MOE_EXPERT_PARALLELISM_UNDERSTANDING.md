# 当前 MoE 专家并行实现学习记录

日期: 2026-05-20

范围: `/aisi-nas/liwentao/deepmd-kit-moe`

## 结论概览

当前专家并行只在 PyTorch DPA3/RepFlow 的 MoE 路径中生效。它不是把整个模型切成多份，而是把 RepFlowLayer Phase 1 的若干 MLP 替换成 MoE:

- M1: `node_self`, 输入 `n_dim`, 输出 `n_dim`
- M2: `node_sym`, 输入 `n_sym_dim`, 输出 `n_dim`
- M3+M4 合并: `edge_experts`, 输入 `edge_info_dim`, 输出 `n_dim + e_dim`
- M5+M7 合并: `angle_experts`, 输入 `angle_dim`, 输出 `e_dim + a_dim`
- M6 仍是普通本地 MLP, 不参与 MoE

核心模块:

- `deepmd/pt/model/descriptor/dpa3.py`: 从线程上下文接收 EP group, 构建 DPA3/RepFlows。
- `deepmd/pt/model/descriptor/repflows.py`: 创建多层 `RepFlowLayer`, 并把 MoE 配置传下去。
- `deepmd/pt/model/descriptor/repflow_layer.py`: 在 `use_moe=True` 时创建 router 和 `MoEDispatchCombine`, forward 时进入 `forward_moe()`。
- `deepmd/pt/model/network/moe_router.py`: 对中心原子 type embedding 做 top-k gating。
- `deepmd/pt/model/network/moe_layer.py`: 完整 dispatch -> expert compute -> combine 流水线。
- `deepmd/pt/model/network/moe_expert.py`: 本地专家权重存储和专家计算。
- `deepmd/pt/model/network/moe_packer.py`: 把 node/edge/angle token 打包成统一宽度, 供 All-to-All 通信使用。
- `deepmd/pt/model/network/moe_ep_ops.py`: 可二阶求导的 All-to-All autograd Function。
- `deepmd/pt/utils/moe_ep_dp.py`: EP/DP group 划分和梯度同步。
- `deepmd/pt/utils/moe_checkpoint.py`: EP checkpoint 的专家参数 gather/slice。

## 配置入口和进程组

训练入口在 `deepmd/pt/entrypoints/main.py`:

1. `dist.init_process_group(backend="cuda:nccl,cpu:gloo")` 先建立默认分布式进程组。
2. `get_trainer()` 检查配置:
   - `model.descriptor.type == "dpa3"`
   - `model.descriptor.repflow.use_moe == true`
   - `training.moe_ep_size > 1`
3. 满足条件后调用 `init_ep_dp_groups(ep_size_config)`, 返回:
   - `ep_group`: 同一个 DP row 内的 EP group, 用于 MoE token All-to-All。
   - `dp_group`: 同一个 EP column 内的 DP group, 用于同一全局专家切片的梯度同步。
   - `ep_rank`, `ep_size`, `dp_rank`, `dp_size`。
4. 这些对象传给 `Trainer(..., use_moe_ep=True, ep_group=..., dp_group=...)`。

`init_ep_dp_groups()` 把 world rank 看成二维网格:

```text
world_size = ep_size * dp_size

例: world_size=4, ep_size=2

          EP rank 0  EP rank 1
DP rank 0: rank 0    rank 1     -> ep_group_0
DP rank 1: rank 2    rank 3     -> ep_group_1
           |         |
           dp_group0 dp_group1
```

这样每个 EP group 负责把 token dispatch 到不同专家所在 GPU；每个 DP group 则持有同一组全局专家切片的副本。

## MoE 上下文如何进入模型

`Trainer.__init__()` 在构造模型前调用 `set_moe_ep_context(ep_group, ep_rank, ep_size)`。这是 thread-local context, 目的是避免把 `ProcessGroup` 放进 config dict, 因为 config 会被 deepcopy/序列化。

随后:

```text
Trainer.__init__
  -> set_moe_ep_context(...)
  -> get_model_for_wrapper(...)
  -> DPA3.__init__
       -> get_moe_ep_context()
       -> DescrptBlockRepflows(..., ep_group, ep_rank, ep_size)
            -> RepFlowLayer(..., ep_group, ep_rank, ep_size)
```

`DPA3.__init__()` 如果拿到非空 `ctx_ep_group`, 会覆盖构造参数中的 `ep_group/ep_rank/ep_size`。因此实际 EP group 是训练器在模型创建前注入的。

## MoE 生效的 RepFlow 条件

`RepFlowLayer.__init__()` 中 `use_moe=True` 时有严格约束:

- `use_dynamic_sel=True`
- `optim_update=False`
- `update_angle=True`
- `n_multi_edge_message=1`
- `a_compress_use_split=True`
- `n_dim:e_dim:a_dim = 4:2:1`

如果不满足, 构造阶段直接报错。原因是当前 MoE 实现绑定了 dynamic selection 的扁平 edge/angle 图索引, 并且 packer 假设固定维度比例来把不同 token 类型打成统一通信行宽。

## 专家参数如何分片

`RepFlowLayer.__init__()` 中:

```python
experts_per_gpu = n_routing_experts // ep_size
```

每个 GPU 只持有 `experts_per_gpu` 个 routing expert。`MoEExpertCollection` 用共享 3D 参数张量存储本地 routing expert:

- `routing_matrix`: `[num_in, num_out, experts_per_gpu]`
- `routing_bias`: `[num_out, experts_per_gpu]`

它还保留 `routing_experts = ModuleList([_ExpertView(...)])` 作为兼容 facade, 但热路径不使用这些子模块。真正计算走 `routing_matrix[:, :, local_eid]` 和 `routing_bias[:, local_eid]`。

shared experts 不切分, 每个 rank 都有完整副本:

```text
routing experts: 按 EP rank 切片
shared experts: 每个 EP rank 完整复制
routers/residual/M6/其他参数: 完整复制
```

全局 expert id 到本地 expert id 的映射:

```text
target_gpu = global_expert_id // experts_per_gpu
local_eid  = global_expert_id % experts_per_gpu
```

## forward 调用总链路

训练时主调用:

```text
Trainer.run.step
  -> self.wrapper(**input_dict, cur_lr=..., label=...)
  -> ModelWrapper.forward
  -> EnergyStdLoss.forward
  -> model(**input_dict)
  -> EnergyModel.forward
  -> forward_common(...)
  -> forward_common_lower(...)
  -> DPAtomicModel.forward_common_atomic(...)
  -> DPA3.forward(...)
  -> DescrptBlockRepflows.forward(...)
  -> RepFlowLayer.forward(...)
  -> RepFlowLayer.forward_moe(...)
  -> MoEDispatchCombine.forward(...)
```

在 `DescrptBlockRepflows.forward()` 中, 每层 `RepFlowLayer` 都会收到 `type_embedding=atype_embd`。`RepFlowLayer.forward()` 发现 `self.use_moe` 后要求 `type_embedding is not None`, 并进入 `forward_moe()`。

## `RepFlowLayer.forward_moe()` 的逻辑

`forward_moe()` 是 MoE 前的特征构建和 MoE 后的物理更新组装。

### Step 1: 构造 MoE 输入

输入张量主要是 dynamic selection 形式:

- `node_ebd_ext`: `[nf, nall, n_dim]`
- `edge_ebd`: `[n_edge, e_dim]`
- `h2`: `[n_edge, 3]`
- `angle_ebd`: `[n_angle, a_dim]`
- `edge_index`: `[2, n_edge]`
- `angle_index`: `[3, n_angle]`
- `type_embedding`: `[nf, nloc, n_dim]`

构造四组 MoE MLP 输入:

1. `node_m1_input = node_ebd.reshape(N_node, n_dim)`
2. `node_m2_input = cat(grrg, drrd).reshape(N_node, n_sym_dim)`
3. `edge_info = cat(node_i, node_j, edge_ebd)`, shape `[N_edge, edge_info_dim]`
4. `angle_info = cat(angle_ebd, node_i_for_angle, edge_k, edge_j)`, shape `[N_angle, angle_dim]`

其中 edge/angle 的路由仍由中心原子决定, 后续用 `n2e_index` 和 `n2a_index` 把 node-level routing broadcast 到 edge/angle token。

### Step 2: 三个独立 router

```python
node_router_out = self.node_router(type_embedding)
edge_router_out = self.edge_router(type_embedding)
angle_router_out = self.angle_router(type_embedding)
```

每个 router 都是 `MLPLayer(input_dim=n_dim, output_dim=n_routing_experts, bias=False)`:

```text
type_embedding [nf, nloc, n_dim]
  -> logits [N_node, n_routing_experts]
  -> torch.topk(logits, k=moe_topk)
  -> softmax(topk_logits)
  -> (topk_weights [N_node, topk], topk_indices [N_node, topk])
```

注意这里 router 输入只用中心原子的 type embedding, 不是 edge/angle 本身的特征。

### Step 3: `MoEDispatchCombine`

`MoEDispatchCombine.forward()` 根据 `ep_group` 分两条路径:

- `ep_group is None`: 单 GPU, 不做 A2A。
- `ep_group is not None`: 多 GPU EP, 做 dispatch/compute/combine。

## 单 GPU MoE 路径

单 GPU路径是 `_forward_single_gpu()`:

1. 对 node router 输出直接处理 M1/M2。
2. 用 `edge_weights_node[n2e_index]`, `edge_indices_node[n2e_index]` 得到 edge token 的 routing。
3. 用 `angle_weights_node[n2a_index]`, `angle_indices_node[n2a_index]` 得到 angle token 的 routing。
4. 对每类 token:
   - 展平成 `[N * topk]`
   - 按 expert id stable sort
   - 按 expert 切 chunk
   - 每个 chunk 用对应本地专家权重矩阵计算
   - scatter 回 `[N, out_dim, topk]`
   - `einsum('ijk,ik->ij', out_3d, weights)` 做 top-k 加权和
5. 加上 shared expert 输出。

单 GPU 路径不会用 `MoEPacker`, 也不会通信。

## 多 GPU EP 路径

多 GPU 路径是 `_forward_multi_gpu()`。完整逻辑如下。

### 1. top-k expand + sort

node 的 M1/M2 输入先拼成:

```text
node_combined = cat(node_m1_input, node_m2_input)  # [N_node, 28a]
```

然后对 node/edge/angle 分别调用 `_topk_expand_sort()` 或 fused CUDA 版本:

```text
features [N, feat_dim]
topk_indices [N, topk]
topk_weights [N, topk]
  -> repeat_interleave features 到 [N*topk, feat_dim]
  -> flat expert ids [N*topk]
  -> 按 global expert id stable sort
  -> sorted_features
  -> sorted_expert_ids
  -> sorted_weights
  -> unsort_idx
  -> counts_per_gpu
```

按 global expert id 排序很关键, 因为:

```text
global_eid = target_gpu * experts_per_gpu + local_eid
```

所以排序后既按目标 GPU 分块, 又保证每个目标 GPU 内部按 local expert id 有序。接收端可以用结构化 O(N) gather, 不再 argsort。

CUDA 优化:

- `USE_FUSED_TOPK_SORT=True`
- CUDA tensor 时走 `fused_topk_expand_sort()`
- 底层 op 是 `torch.ops.deepmd.moe_topk_expand_sort`
- C++ wrapper: `source/op/pt/moe_topk_expand_sort.cc`
- CUDA kernel: `source/lib/src/gpu/moe_topk_expand_sort.cu`

### 2. pack for dispatch

`MoEPacker` 把三种不同宽度的 token 打成统一行宽, 减少 A2A 次数。

输入 packing 规则, 令 `a = a_dim`:

```text
Node:  [N, 28a] -> pad 到 [N, 40a]             每个 node 一行
Edge:  [N, 10a] -> 每 4 个 concat 成 [ceil(N/4), 40a]
Angle: [N,  4a] -> 每 10 个 concat 成 [ceil(N/10), 40a]
```

对每个目标 GPU g, pack 顺序是:

```text
node rows for g
edge rows for g
angle rows for g
```

返回:

- `packed`: `[total_send_rows, 40a]`
- `send_splits`: 每个目标 GPU 要发送的 packed row 数

CUDA 优化:

- `USE_FUSED_PACK=True`
- CUDA tensor 时走 `fused_pack_for_dispatch()`
- 底层 op 是 `torch.ops.deepmd.moe_pack_for_dispatch`
- C++ wrapper: `source/op/pt/moe_pack_for_dispatch.cc`
- CUDA kernel: `source/lib/src/gpu/moe_pack_for_dispatch.cu`

### 3. metadata exchange

数据 A2A 前必须知道每个 rank 会收到多少 node/edge/angle token。

当前 rank 构造:

```text
send_info[g] = (node_counts[g], edge_counts[g], angle_counts[g])
```

`exchange_metadata(send_info, ep_group)` 用 `dist.all_to_all()` 交换整数元数据。返回:

```text
recv_info[g] = rank g 将发给当前 rank 的 (node_count, edge_count, angle_count)
```

再通过 `counts_to_packed_rows()` 得到 `recv_splits`。

### 4. dispatch A2A

```python
recv_tensor = all_to_all_differentiable(
    packed, send_splits, recv_splits, ep_group
)
```

`all_to_all_differentiable()` 使用 `_AllToAllDouble`:

- forward 调 `dist.all_to_all_single()`
- backward 再调用同一个 autograd Function, 交换 send/recv splits
- 这样 `create_graph=True` 的二阶导数也能跨 A2A 边界传播, 满足 DeePMD force/virial 的高阶梯度需求

### 5. unpack + expert id exchange

`packer.unpack_from_dispatch()` 把 `[recv_rows, 40a]` 拆回:

- `node_recv`: `[sum(recv_node_counts), 28a]`
- `edge_recv`: `[sum(recv_edge_counts), 10a]`
- `angle_recv`: `[sum(recv_angle_counts), 4a]`

expert id 是整数, 不需要梯度。默认优化路径 `_exchange_expert_ids_batched()` 把 node/edge/angle 的 expert id 合并成一次 `dist.all_to_all_single()`, 收到后再拆回:

```text
node_eid_recv
edge_eid_recv
angle_eid_recv
```

### 6. 本地 expert compute

接收端已经只收到属于本 rank 专家的 token。计算前:

```text
local_eids = expert_ids % experts_per_gpu
```

`_build_expert_gather_idx()` 利用每个 sender segment 已按 local expert id 有序的事实, 生成:

- `gather_idx`: 把 recv buffer 重排成按本地专家连续
- `split_sizes`: 每个本地专家 token 数
- `ungather_idx`: 计算后恢复 recv 顺序

node 同时算 M1/M2:

```text
node_recv -> split node_m1_recv / node_m2_recv
  -> _compute_node_experts()
       -> gather
       -> node_self_experts.forward_expert_batched()
       -> node_sym_experts.forward_expert_batched()
       -> ungather
```

edge/angle 类似:

```text
_compute_feature_experts()
  -> gather
  -> expert_collection.forward_expert_batched()
  -> ungather
```

`forward_expert_batched()` 使用 `routing_matrix.permute(2, 0, 1)` 得到 `[E, I, O]`, 对每个 expert chunk 做 matmul。`USE_ULTRA_OPTIMIZED=True` 时尝试使用 `moe_ultra_optimized.batched_expert_forward_optimized()`, 在负载相对均衡时走 padding + `torch.bmm`, 否则回退 per-expert loop。

### 7. pack for combine + 反向 A2A

专家输出打包成输出行宽 `30a`:

```text
Node:  [N, 8a] -> pad 到 [N, 30a]
Edge:  [N, 6a] -> 每 4 个 concat, 再 pad 到 30a
Angle: [N, 3a] -> 每 10 个 concat 成 30a
```

然后反向 A2A:

```python
returned = all_to_all_differentiable(
    packed_out, recv_splits, send_splits, ep_group
)
```

注意 splits 交换了, 这样结果回到 token 原始 owner rank。

### 8. unpack, unsort, top-k weighted sum

回到 owner rank 后:

1. `unpack_from_combine()` 拆出 node/edge/angle 输出。
2. 用 `unsort_idx` 恢复 top-k 展开前的原始 token 顺序。
3. `_weighted_sum_topk()` 把 `[N*topk, out_dim]` reshape 为 `[N, topk, out_dim]`, 用 router weight 加权求和。
4. 加上 shared expert 输出。

shared expert 在 `USE_ULTRA_OPTIMIZED=True` 且 CUDA 可用时会在单独 CUDA stream 上提前计算, 用来和 expert id exchange / expert compute 重叠。

## MoE 输出如何回到 RepFlow 更新

`MoEDispatchCombine` 返回:

```text
node_m1_out:      [N_node, n_dim]
node_m2_out:      [N_node, n_dim]
edge_merged_out:  [N_edge, n_dim + e_dim]
angle_merged_out: [N_angle, e_dim + a_dim]
```

`RepFlowLayer.forward_moe()` 接着:

1. `edge_merged_out.split([n_dim, e_dim])` 得到:
   - `node_edge_out` 对应 M3
   - `edge_self_out` 对应 M4
2. `angle_merged_out.split([e_dim, a_dim])` 得到:
   - `edge_angle1_out` 对应 M5
   - `angle_self_out` 对应 M7
3. `node_edge_out * sw` 按 `n2e_index` reduce 回 node。
4. `edge_angle1_out * a_sw` 按 `eij2a_index` reduce 回 edge。
5. M6: `edge_angle_linear2_moe(edge_angle_agg)` 本地普通 MLP。
6. `list_update()` 组合 node/edge/angle 残差更新。

## 反向传播和梯度同步

由于 DDP 默认会把所有参数梯度按 world group all-reduce, 但 MoE routing expert 参数只在某些 EP rank 上存在, 所以 `Trainer.run.step()` 在 `use_moe_ep=True` 时:

```python
with self.wrapper.no_sync():
    loss.backward()
sync_moe_gradients(...)
```

`sync_moe_gradients()` 的规则:

- routing expert 参数:
  - 参数名包含 `.routing_matrix`, `.routing_bias`, 或旧格式 `.routing_experts.`
  - 只在 `dp_group` 内 all-reduce
  - 除以 `world_size`, 因为 EP A2A backward 已经聚合了同一个 EP group 内 token 路径的梯度贡献
- 非 routing expert 参数:
  - router、shared experts、普通 MLP、残差等
  - 在 world group all-reduce
  - 除以 `world_size`
- `dp_size == 1` 时:
  - routing expert 不额外 all-reduce
  - 非 routing 参数仍做 world all-reduce

## checkpoint 保存和加载

EP 下每个 rank 只有一部分 routing experts。保存时 `Trainer.save_model()` 调用:

```python
moe_state_dict_to_global(module, ep_rank, ep_size, experts_per_gpu, ep_group)
```

对 `.routing_matrix` / `.routing_bias`:

```text
rank local tensor:
  routing_matrix [I, O, experts_per_gpu]
  routing_bias   [O, experts_per_gpu]

all_gather over ep_group
  -> cat on expert dim
  -> global:
       routing_matrix [I, O, n_routing_experts]
       routing_bias   [O, n_routing_experts]
```

非 routing 参数直接保存当前 rank 的副本。加载时 `moe_load_state_dict_from_global()` 按:

```text
start = ep_rank * experts_per_gpu
end = start + experts_per_gpu
```

从 global tensor 最后一维切出本 rank 的专家切片。

## 当前实现的关键假设和风险点

1. `n_routing_experts` 必须能被 `ep_size` 整除。代码直接 `n_routing_experts // ep_size`, 没看到 RepFlowLayer 内部显式校验余数, 依赖配置正确性。
2. MoE 只支持 DPA3/RepFlow 的 dynamic selection 路径, 不是通用 descriptor MoE。
3. edge/angle routing 是从中心 node routing broadcast 出来的, 不是按 edge/angle feature 独立决策。
4. `MoEPacker` 强依赖 `n_dim:e_dim:a_dim = 4:2:1`。
5. `all_to_all_differentiable()` 是为了二阶梯度设计的, 不能随意换成普通 `dist.all_to_all_single()` 包装, 否则 force/virial 梯度链会断。
6. DDP 自动同步在 MoE EP 下必须关闭, 否则 routing expert 梯度会在错误 group 上同步。

## 一句话调用图

```text
torchrun + dp --pt train
  -> dist.init_process_group
  -> init_ep_dp_groups(training.moe_ep_size)
  -> Trainer(set_moe_ep_context)
  -> DPA3/RepFlows/RepFlowLayer 构造 MoE
  -> RepFlowLayer.forward_moe
  -> router(type_embedding)
  -> topk expand/sort
  -> pack node/edge/angle
  -> differentiable A2A dispatch
  -> local expert compute
  -> differentiable A2A combine
  -> unsort + weighted sum + shared experts
  -> RepFlow residual updates
  -> loss.backward(no_sync)
  -> sync_moe_gradients
  -> optimizer.step
```
