# deepmd-kit-arch 图并行实现学习记录

日期: 2026-05-21

阅读对象: `/aisi-nas/liwentao/deepmd-kit-arch`

记录位置: `/aisi-nas/liwentao/deepmd-kit-moe/history`

说明: `/aisi-nas/liwentao/deepmd-kit-arch` 目录下还存在一个嵌套的 `deepmd-kit-arch/` 副本。本记录以仓库根目录实现为准, 因为根目录包含当前图并行启动脚本、`graph_parallel.py`、`distutils.py` 和已有 GP 文档。

## 总结

`deepmd-kit-arch` 的图并行不是把一个完整 DeePMD 模型复制成多个子模型, 也不是先调用通用图划分库做复杂 graph partition。

它的当前实现是:

1. 用 `torchrun + DeviceMesh` 建立 `(PP, DP, EP, GP)` 拓扑。
2. 用 `GP` 维表示图并行组。
3. DataLoader 和 DDP 只看 DATA group, 让同一个 GP group 内的 ranks 拿同一个 datapoint。
4. 在 DPA3/RepFlow 的 dynamic graph 里, 把展平后的 node 按连续区间均衡切成 `gp_world_size` 份。
5. edge 和 angle 按中心 node 归属进入对应 partition。
6. 每个 RepFlow layer 中, 每个 GP rank 只算自己的 edge/angle 分区。
7. 非最后一层 gather 回完整 node embedding, 让下一层继续有全图 node 信息。
8. 最后一层保留 node descriptor 分片, 上层 fitting/output 再按 partition 处理和合并。
9. 能量、force、virial 等 frame/global 量通过 GP group reduce; 原子级输出通过 GP gather 拼回。

一句话:

```text
Data/DP 负责不同 datapoint;
GP 负责同一个 datapoint 内按 node partition 分摊 RepFlow 图计算。
```

## 关键文件

图并行相关文件集中在这些位置:

```text
/aisi-nas/liwentao/deepmd-kit-arch/tools/run_gp_pt_train.py
/aisi-nas/liwentao/deepmd-kit-arch/run_2rank_gp.sh
/aisi-nas/liwentao/deepmd-kit-arch/run_4rank_single_gpu_gp.sh
/aisi-nas/liwentao/deepmd-kit-arch/run_single_process_gp_loop.sh
/aisi-nas/liwentao/deepmd-kit-arch/distutils.py
/aisi-nas/liwentao/deepmd-kit-arch/graph_parallel.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/utils/local_distutils.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/utils/dataloader.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/train/training.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/model/model/make_model.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/model/descriptor/dpa3.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/model/descriptor/repflows.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/model/atomic_model/dp_atomic_model.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/model/atomic_model/base_atomic_model.py
/aisi-nas/liwentao/deepmd-kit-arch/deepmd/pt/model/model/transform_output.py
```

其中真正切图的位置是:

```text
deepmd/pt/model/descriptor/repflows.py
```

通信抽象在:

```text
graph_parallel.py
distutils.py
```

训练侧 DP/GP 同步在:

```text
deepmd/pt/train/training.py
deepmd/pt/utils/dataloader.py
```

## 并行拓扑

当前设计把 world ranks 看成四维:

```text
(PP, DP, EP, GP)
```

常见配置有两种。

### 纯图并行

`run_2rank_gp.sh` 默认:

```text
PP=1
DP=1
EP=1
GP=2
WORLD_SIZE=2
```

语义:

```text
rank 0 和 rank 1 拿同一个 datapoint;
rank 0 计算该图的 GP partition 0;
rank 1 计算该图的 GP partition 1;
两个 rank 共同组成一个 GP group。
```

### DP x GP

`run_4rank_single_gpu_gp.sh` 默认:

```text
PP=1
DP=2
EP=1
GP=2
WORLD_SIZE=4
```

逻辑上:

```text
dp=0: gp ranks [0, 1]  -> 同一个 datapoint A
dp=1: gp ranks [2, 3]  -> 同一个 datapoint B
```

GP groups:

```text
[0, 1]
[2, 3]
```

DATA/DDP groups:

```text
[0, 2]  # gp_rank=0 的 DP replicas
[1, 3]  # gp_rank=1 的 DP replicas
```

所以:

- GP group 内 ranks 共享同一个样本, 分摊一个图。
- DATA group 内 ranks 做数据并行, 处理不同样本但同一个 GP shard。

## 启动入口

### shell 脚本

两卡纯 GP:

```bash
cd /aisi-nas/liwentao/deepmd-kit-arch
./run_2rank_gp.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json
```

四卡 DP x GP:

```bash
cd /aisi-nas/liwentao/deepmd-kit-arch
./run_4rank_single_gpu_gp.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json
```

单进程模拟:

```bash
cd /aisi-nas/liwentao/deepmd-kit-arch
./run_single_process_gp_loop.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json
```

这些脚本最终都进入:

```text
tools/run_gp_pt_train.py
```

### 关键环境变量

```text
DISABLE_GP_MODE
  "1" 表示禁用 GP, 走原始全图路径。

DP_GP_EXEC_MODE
  distributed          真正多进程 GP。
  single-process-loop  单进程 for-loop 模拟 GP。
  auto                 优先真实 GP, 不满足条件时回退。

DP_ENABLE_DISTUTILS_GP
  是否允许 tools/run_gp_pt_train.py 初始化仓库自己的 distutils GP 后端。

DP_GP_SIZE
  GP 维大小。

DP_DP_SIZE
  DP 维大小。

DP_PP_SIZE
  PP 维大小。

DP_EP_SIZE
  EP 维大小。

DP_GP_SIM_WORLD_SIZE
  single-process-loop 模式下模拟几个 GP partition。

DP_PT_DIST_BACKEND
  DeePMD fallback 初始化普通 torch.distributed 时用的 backend。

DP_DEBUG_2X2
  打印 2x2 DP x GP 拓扑和分区调试信息。
```

## `tools/run_gp_pt_train.py` 的调用逻辑

`tools/run_gp_pt_train.py` 是推荐入口。它做的是:

```text
tools/run_gp_pt_train.py
  -> 把 repo root 插入 sys.path
  -> get_repo_distutils()
  -> _maybe_setup_distributed_gp()
  -> deepmd_main(["--pt", ...])
```

`_maybe_setup_distributed_gp()` 的判断逻辑:

```text
if DISABLE_GP_MODE == "1":
    不初始化 GP

if DP_ENABLE_DISTUTILS_GP != "1":
    不初始化 GP

if WORLD_SIZE <= 1:
    不初始化 GP

if DP_GP_EXEC_MODE 是 single-process-loop:
    不初始化分布式 GP

否则:
    读取 DP_GP_SIZE / DP_DP_SIZE / DP_PP_SIZE / DP_EP_SIZE
    校验 PP * DP * EP * GP == WORLD_SIZE
    调用 distutils.setup(...)
```

然后它设置:

```python
cli_args = ["--pt", *sys.argv[1:]]
deepmd_main(cli_args)
```

所以用户仍然看起来是在跑 DeePMD PyTorch 训练, 只是外层提前把 GP group 初始化好了。

## 为什么要 `local_distutils.py`

Python 有标准库/第三方 `distutils`, 仓库根目录也有一个 `distutils.py`。为了避免 import 错模块, 代码通过:

```text
deepmd/utils/local_distutils.py
```

显式加载:

```text
/aisi-nas/liwentao/deepmd-kit-arch/distutils.py
```

并注册为:

```text
_deepmd_repo_distutils
```

这样 `graph_parallel.py`、`training.py`、`dataloader.py` 都能拿到同一个仓库版分布式工具模块。

## `distutils.py` 如何建立 GP group

入口:

```python
distutils.setup(...)
```

调用链:

```text
distutils.setup
  -> 读取 LOCAL_RANK / RANK / WORLD_SIZE
  -> torch.cuda.set_device(local_rank)
  -> dist.init_process_group(...)
  -> setup_dist_group(...)
```

`setup_dist_group()` 在非 FSDP 情况下建立 DeviceMesh:

```python
mesh = init_device_mesh(
    device_type,
    (pp_size, dp_size, ep_size, gp_size),
    mesh_dim_names=("pp", "dp", "ep", "gp"),
)
```

然后抽出:

```text
_PP_GROUP = mesh.get_group("pp")
_DP_GROUP = mesh.get_group("dp")
_EP_GROUP = mesh.get_group("ep")
_GP_GROUP = mesh.get_group("gp")
```

还会尝试 flatten 出复合 group:

```text
_DP_GP_GROUP
_DP_GP_EP_GROUP
_DATA_GROUP = dp_ep
```

其中 DATA group 用于数据切分和 DDP:

```text
Data group = DP x EP, 不包含 GP
```

这点是 DP x GP 能正确工作的关键。GP ranks 不能被当成不同数据 replica, 否则同一个图会被不同 GP rank 拿到不同样本。

## `graph_parallel.py` 的封装

业务代码不直接访问 `distutils.py`, 而是用:

```text
graph_parallel.py
```

它提供:

```text
graph_parallel_enabled()
get_gp_rank()
get_gp_world_size()
gather_node_tensor()
gather_node_tensor_no_sum_grad()
reduce_graph_tensor()
_balanced_partition_sizes()
_build_partition_offsets()
```

### rank/world 获取

优先级:

```text
1. 如果仓库 distutils 已初始化, 用 distutils 的 GP group。
2. 否则如果 torch.distributed 已初始化, 用普通 dist world。
3. 否则用 mock rank/world size, 供单进程模拟或测试。
```

### `gather_node_tensor()`

语义:

```text
forward:  gather GP group 中每个 rank 的 node shard, cat 成完整 node tensor。
backward: 对完整 gradient 做 GP all-reduce, 然后 split 回各 rank。
```

底层优先调用:

```python
distutils.gather_from_model_parallel_region_sum_grad(...)
```

用途:

```text
RepFlow 中间层之后把本地 node embedding 拼回整图。
```

为什么 backward 要 sum grad:

```text
中间层 gather 后, 下一层每个 GP rank 都看到完整 node embedding。
同一个 node 的梯度可能来自多个 GP rank 的局部边/角计算。
因此反向时要先在 GP group 里求和, 再把属于本 rank 的 shard 切回去。
```

### `gather_node_tensor_no_sum_grad()`

语义:

```text
forward: gather + cat。
backward: 只 split, 不做额外 GP all-reduce。
```

底层优先调用:

```python
distutils.gather_from_model_parallel_region(...)
```

用途:

```text
输出组装阶段拼原子级输出或 mask。
```

这里不能 sum grad, 因为它只是把互不重叠的原子分片拼起来, 不是多个 rank 共同贡献同一个 node 的中间表示。

### `reduce_graph_tensor()`

语义:

```text
GP group all-reduce sum。
```

底层优先调用:

```python
distutils.reduce_from_model_parallel_region(...)
```

用途:

```text
frame 级 energy_redu
force / virial / virial_redu 等需要把各 GP shard 贡献求和的量。
```

## DataLoader 如何适配 DP x GP

文件:

```text
deepmd/pt/utils/dataloader.py
```

关键函数:

```python
_get_data_parallel_rank_world()
```

优先从仓库版 distutils 读取:

```python
data_rank = gp_distutils.get_data_rank()
data_world_size = gp_distutils.get_data_world_size()
```

然后 `DistributedSampler` 使用的是:

```python
DistributedSampler(
    system,
    num_replicas=data_world_size,
    rank=data_rank,
)
```

不是 global rank/world size。

在 `DP=2, GP=2` 下:

```text
rank 0: data_rank=0, gp_rank=0
rank 1: data_rank=0, gp_rank=1
rank 2: data_rank=1, gp_rank=0
rank 3: data_rank=1, gp_rank=1
```

因此:

```text
rank 0/1 拿同一个样本, 两个 GP shard 共同处理。
rank 2/3 拿另一个样本, 两个 GP shard 共同处理。
```

如果这里用 global rank 做 sampler, GP 就会错, 因为同一个 GP group 内各 rank 会拿不同图。

## Trainer 如何使用 DATA group 和 GP group

文件:

```text
deepmd/pt/train/training.py
```

### rank 辅助函数

训练器里有:

```python
_get_data_parallel_rank()
_get_graph_parallel_rank()
_get_data_parallel_world_size()
_get_data_parallel_group()
```

它们优先使用仓库版 distutils, 失败时才回退到普通 `torch.distributed`。

### DDP 包裹

Trainer 初始化 wrapper 后:

```python
data_world_size = _get_data_parallel_world_size()
if dist initialized and data_world_size > 1:
    data_group = _get_data_parallel_group()
    self.wrapper = DDP(..., process_group=data_group)
else:
    _maybe_sync_graph_parallel_parameters(self.wrapper)
```

含义:

- DP x GP: DDP 只在 DATA group 内同步, 不跨 GP group。
- 纯 GP: 没有 DDP, 只在训练开始时用 GP group broadcast 参数, 保证各 GP rank 初始参数一致。

### backward 后的 GP 梯度同步

训练循环中:

```python
loss.backward()
_maybe_sync_graph_parallel_gradients(self.wrapper)
optimizer.step()
```

`_maybe_sync_graph_parallel_gradients()` 调用:

```python
gp_distutils.allreduce_gradients(
    actual_model,
    dp_group=gp_distutils.get_gp_group(),
    world_size=gp_distutils.get_gp_world_size(),
    average=False,
)
```

注意 `average=False`。

原因:

- DDP 已经在 DATA group 内做数据并行平均。
- GP group 内是同一个样本被拆成多个 shard, 梯度语义是累加同一图的各 shard 贡献, 所以是 sum, 不是 average。

## 普通 DeePMD forward 到 GP forward 的调用链

训练时主链路:

```text
Trainer.run()
  -> Trainer.get_data()
  -> ModelWrapper.forward(...)
  -> EnergyStdLoss.forward(...)
  -> model(**input_dict)
  -> EnergyModel.forward(...)
  -> make_model.CM.forward_common(...)
```

`forward_common()` 先走标准 DeePMD 扩展邻域:

```text
input coord/atype/box
  -> input_type_cast
  -> extend_input_and_build_neighbor_list(...)
       extended_coord
       extended_atype
       mapping
       nlist
  -> forward_common_lower(...)
```

也就是说, GP 分区不是在 DataLoader 阶段做, 也不是在 neighbor list 构建阶段做。每个 GP rank 先拿到同一个样本和完整 neighbor list, 然后在 RepFlow descriptor 内部切图。

## `make_model.forward_common_lower()` 的 GP 分支

文件:

```text
deepmd/pt/model/model/make_model.py
```

调用:

```python
atomic_ret = self.atomic_model.forward_common_atomic(
    cc_ext,
    extended_atype,
    nlist,
    mapping=mapping,
    fparam=fp,
    aparam=ap,
    comm_dict=comm_dict,
)
```

如果:

```python
atomic_ret.get("gp_mode", False)
```

则进入 GP 输出合并逻辑。

核心步骤:

1. 取出 `fit_ret_parts` 和 `mask_parts`。
2. 对每个 partition 的 `local_fit_ret` 单独调用 `fit_output_to_model_output(...)`。
3. 原子级输出和 mask 在本 rank 内 concat。
4. 如果是真实 distributed GP, 用 `gather_node_tensor_no_sum_grad(...)` 跨 GP ranks 拼回完整原子输出。
5. 对 reducible 量, 例如 energy/force/virial, 先 stack 本地 parts 求和。
6. 如果是真实 distributed GP, 用 `reduce_graph_tensor(...)` 在 GP group 内求和。

伪调用:

```text
for local_fit_ret in fit_ret_parts:
    local_model_predict = fit_output_to_model_output(local_fit_ret, ...)
    model_predict_parts.append(local_model_predict)

atom outputs:
    concat local parts
    if distributed GP:
        gather_node_tensor_no_sum_grad(...)

reduced outputs:
    sum local parts
    if distributed GP:
        reduce_graph_tensor(...)
```

如果不是 GP mode, 就走原始:

```python
fit_output_to_model_output(atomic_ret, ...)
```

最后 `forward_common()` 还会调用:

```python
communicate_extended_output(...)
```

把 extended atom 域的导数通过 `mapping` scatter/reduce 回 local atom 域。

## DPA3 如何识别 GP 输出

文件:

```text
deepmd/pt/model/descriptor/dpa3.py
```

DPA3 正常调用:

```python
repflows_output = self.repflows(
    nlist,
    extended_coord,
    extended_atype,
    node_ebd_ext,
    mapping,
    comm_dict=comm_dict,
)
```

然后判断:

```python
is_gp_mode = isinstance(repflows_output, dict) and "gp_partitions" in repflows_output
```

如果是 GP mode:

- 不把 `node_ebd_parts` cat 成普通 descriptor。
- 只做 dtype cast。
- 保留 dict 结构直接返回给 AtomicModel。

如果不是 GP mode:

- 返回普通五元组:

```text
descriptor, rot_mat, edge_ebd, h2, sw
```

## RepFlows 中图是如何构建成 dynamic graph 的

文件:

```text
deepmd/pt/model/descriptor/repflows.py
```

RepFlows 输入:

```text
nlist             [nf, nloc, nnei]
extended_coord    [nf, nall, 3]
extended_atype    [nf, nall]
extended_atype_embd
mapping
```

前半部分是标准 DPA3/RepFlow:

1. 先用 `emask` 排除不合法 pair。
2. 用 `prod_env_mat(...)` 生成 edge 环境:

```text
dmatrix, diff, sw
```

3. 根据 `a_rcut/a_sel` 从 edge neighbor 中筛出 angle neighbor:

```text
a_nlist
a_diff
a_sw
```

4. 把 padding neighbor 的 index 从 `-1` 替换为 `0`, 同时用 mask 保存真实性。
5. 如果 `not parallel_mode and use_loc_mapping`, 用 `mapping` 把 `nlist` 从 extended index 转成 local index。
6. 如果 `use_dynamic_sel=True`, 调:

```python
node_index, edge_index, angle_index = get_graph_index(...)
```

`get_graph_index()` 来自:

```text
deepmd/pt/model/network/utils.py
```

返回:

```text
node_index   [nf * nloc]
edge_index   [2, n_edge]
angle_index  [3, n_angle]
```

其中:

```text
edge_index[0] = edge owner / center node index
edge_index[1] = neighbor node index

angle_index[0] = angle owner / center node index
angle_index[1] = edge ij id
angle_index[2] = edge ik id
```

随后动态选择会把 edge/angle tensor 压成真正的 flat token:

```text
edge_input [n_edge, ...]
h2         [n_edge, 3]
sw         [n_edge]
angle_input[n_angle, ...]
a_sw       [n_angle]
```

图并行就是在这个 dynamic graph 上做 partition。

## RepFlows 如何决定是否启用 GP

RepFlows 中的开关:

```python
gp_enabled = os.environ.get("DISABLE_GP_MODE", "0") != "1"
gp_exec_mode = os.environ.get("DP_GP_EXEC_MODE", "auto").strip().lower()
```

然后检查:

```text
graph_parallel.graph_parallel_enabled()
graph_parallel._distutils_gp_ready()
```

模式大致分为:

```text
distributed:
  需要 distutils GP backend ready。
  每个 rank 只处理自己的 partition。

single-process-loop:
  不需要分布式后端。
  单进程循环所有 partition。

auto:
  如果真实 GP backend 可用, 走 distributed。
  否则用 mock/sim world size 做本地模拟。
```

真正进 GP 分支还要求:

```text
gp_enabled == True
self.use_dynamic_sel == True
not is_debug
```

也就是说当前 GP 绑定 DPA3/RepFlow dynamic selection 路径。关闭 dynamic selection 或换普通 descriptor, 基本不会进入这套图切分。

## GP partition 如何构造

核心逻辑:

```python
gp_num_nodes = node_ebd.shape[0]
gp_sizes = graph_parallel._balanced_partition_sizes(gp_num_nodes, gp_world_size)
gp_offsets = graph_parallel._build_partition_offsets(gp_sizes, node_ebd.device)
```

如果:

```text
gp_num_nodes = 10
gp_world_size = 4
```

则:

```text
gp_sizes = [3, 3, 2, 2]
offsets  = [0, 3, 6, 8]
```

每个 rank 的节点范围:

```text
rank0: [0, 3)
rank1: [3, 6)
rank2: [6, 8)
rank3: [8, 10)
```

当前切分策略是:

```text
按 flattened node id 的连续区间均衡切分。
```

不是 METIS, 不是最小边割, 也不按空间域切分。

### edge 归属

edge 跟随中心 node:

```python
edge_mask = (edge_index[0] >= local_start) & (edge_index[0] < local_end)
local_edge_index = edge_index[:, edge_mask]
```

含义:

```text
只要 edge owner node 属于当前 partition, 这个 edge 就属于当前 partition。
```

### angle 归属

angle 也跟随中心 node:

```python
angle_mask = (angle_index[0] >= local_start) & (angle_index[0] < local_end)
local_angle_ids = nonzero(angle_mask)
local_angle_index = angle_index[:, local_angle_ids].clone()
```

### angle 中 edge id 的 remap

angle_index 的第 1/2 行引用的是 edge id。

切分 edge 后, 当前 partition 内的 edge id 已经不是全局 edge id。因此代码会构造:

```python
edge_remap = torch.full((num_total_edges,), -1)
edge_remap[edge_mask] = torch.arange(local_edge_count)
```

然后:

```python
local_angle_index[1] = edge_remap[local_angle_index[1]]
local_angle_index[2] = edge_remap[local_angle_index[2]]
```

并过滤掉:

```text
引用非法 edge id 的 angle
引用不属于本地 edge 集合的 angle
```

最终每个 partition 存:

```text
rank
local_start
local_end
local_size
edge_mask
edge_index
angle_mask
angle_index
```

这就是 GP 分区元数据 `gp_partitions`。

## RepFlow layer 在 GP 下如何执行

RepFlows 会遍历每个 `RepFlowLayer`:

```python
for idx, ll in enumerate(self.layers):
    ...
```

### distributed GP

真实多进程 GP:

```text
gp_rank = graph_parallel.get_gp_rank()
partition = gp_partitions[gp_rank]
```

当前 rank 只取自己的局部数据:

```text
curr_edge_ebd = edge_ebd[local_edge_mask]
curr_angle_ebd = angle_ebd[local_angle_mask]
local_h2 = h2[local_edge_mask]
local_sw = sw[local_edge_mask]
local_a_sw = a_sw[local_angle_mask]
```

然后调用:

```python
node_ebd_out, ret_edge_ebd, ret_angle_ebd = ll.forward(
    node_ebd,
    curr_edge_ebd,
    local_h2,
    curr_angle_ebd,
    local_sw,
    local_a_sw,
    edge_index=local_edge_index,
    angle_index=local_angle_index,
)
```

注意这里传入的是:

```text
完整 node_ebd + 局部 edge/angle
```

不是只传局部 node。这样本 partition 的 edge 如果指向远端 node, 仍然能通过 `edge_index[1]` 访问完整 node embedding。

layer 返回后只保留当前 rank 负责的 node slice:

```python
node_ebd_local = node_ebd_out[local_start:local_end]
```

如果不是最后一层:

```python
node_ebd = graph_parallel.gather_node_tensor(node_ebd_local, dim=0)
```

也就是把所有 GP rank 的 node slice all-gather 成完整 node embedding, 供下一层继续用。

如果是最后一层:

```python
node_ebd = [node_ebd_local]
```

不再 gather 成完整 tensor, 而是把本地分片交给上层 fitting/output。

### single-process-loop

单进程模拟时:

```text
for partition in gp_partitions:
    切局部 edge/angle
    调 ll.forward(...)
    收集 node_ebd_local
```

非最后一层:

```python
node_ebd = torch.cat(node_ebd_parts, dim=0)
```

最后一层:

```python
node_ebd = node_ebd_parts
```

所以它是 distributed GP 的串行等价模拟版, 常用于排除通信问题, 单独验证切图逻辑。

## RepFlows 返回给 DPA3 的 GP dict

最后一层结束后, 如果 GP 生效且 `node_ebd` 是 list, RepFlows 返回:

```python
{
    "gp_partitions": gp_partitions,
    "local_partition_indices": local_partition_indices,
    "node_ebd_parts": node_ebd,
    "node_index_parts": node_index_parts,
    "rot_mat_parts": [None] * len(selected_partitions),
    "h2_parts": h2_parts,
    "sw_parts": sw_parts,
}
```

distributed GP 下:

```text
local_partition_indices = [graph_parallel.get_gp_rank()]
node_ebd_parts = [本 rank 的 node descriptor]
```

single-process-loop 下:

```text
local_partition_indices = [0, 1, ..., gp_world_size-1]
node_ebd_parts = 所有 partition 的 node descriptor 列表
```

## AtomicModel 如何处理 GP dict

文件:

```text
deepmd/pt/model/atomic_model/dp_atomic_model.py
```

`DPAtomicModel.forward_atomic()` 调 descriptor 后检查:

```python
is_gp_mode = isinstance(descriptor_output, dict) and "gp_partitions" in descriptor_output
```

如果是 GP mode:

1. 取出:

```text
gp_partitions
local_partition_indices
node_ebd_parts
node_index_parts
rot_mat_parts
h2_parts
```

2. 把 `atype` 展平成:

```python
atype_flat = atype.reshape(-1)
```

3. 对每个本地 partition:

```python
partition = gp_partitions[partition_index]
local_start = partition["local_start"]
local_end = partition["local_end"]

local_descriptor = node_ebd_parts[i]
local_atype = atype_flat[local_start:local_end]
local_aparam = aparam_flat[local_start:local_end] if aparam is not None else None
local_h2 = h2_parts[i]
local_batch = node_index_parts[i]
```

4. 调 fitting net:

```python
local_fit_ret = self.fitting_net(
    local_descriptor,
    local_atype,
    gr=local_rot_mat,
    g2=None,
    h2=local_h2,
    fparam=fparam,
    aparam=local_aparam,
)
```

5. 给每个 local fit result 加:

```text
batch
n_batch
```

6. 返回:

```python
{
    "fit_ret_parts": fit_ret_parts,
    "gp_partitions": gp_partitions,
    "local_partition_indices": local_partition_indices,
    "gp_mode": True,
}
```

因此 GP 不只切 descriptor; fitting net 也按 partition 分块执行。

## BaseAtomicModel 如何应用 out_stat 和 mask

文件:

```text
deepmd/pt/model/atomic_model/base_atomic_model.py
```

`forward_common_atomic()` 在 `ret_dict["gp_mode"] == True` 时:

1. 同步 GP out stat:

```python
self._sync_out_stat_for_gp()
```

2. 构造完整 atom mask:

```python
atom_mask = ext_atom_mask[:, :nloc].to(torch.int32)
```

3. 展平:

```python
atype_flat = atype.reshape(-1)
atom_mask_flat = atom_mask.reshape(-1)
```

4. 对每个 local partition:

```text
local_atype = atype_flat[local_start:local_end]
local_mask = atom_mask_flat[local_start:local_end]
local_fit_ret = apply_out_stat(fit_ret_parts[i], local_atype)
local_fit_ret *= local_mask
```

5. 返回更新后的:

```text
fit_ret_parts
mask_parts
local_partition_indices
```

这一步保证每个 shard 的输出统计和虚原子 mask 都和 partition 对齐。

## 输出如何合并

GP 输出合并主要在:

```text
deepmd/pt/model/model/make_model.py
```

`transform_output.py` 里也保留了一个 GP 分支, 但根目录当前主路径以 `make_model.py` 的 `atomic_ret["gp_mode"]` 分支为主。

### 原子级输出

对每个变量 `kk`, 先收集本 rank 处理的 local parts:

```text
local_atomic_values = [part[kk] for part in model_predict_parts]
```

如果是单进程 loop:

```text
cat 本进程所有 partition
```

如果是 distributed GP:

```python
local_atomic_tensor = graph_parallel.gather_node_tensor_no_sum_grad(...)
```

因为原子级输出是不同 node shard 的拼接, 不应该在 backward 里对同一个位置求和。

### reduced 输出

例如 energy:

```text
energy_redu = sum(local energy_redu parts)
```

distributed GP 下再:

```python
energy_redu = graph_parallel.reduce_graph_tensor(energy_redu)
```

force、virial、virial_redu 这类由局部贡献组成的量也走类似 reduce 语义。

### extended -> local

最后 `forward_common()` 调:

```python
communicate_extended_output(
    model_predict_lower,
    self.model_output_def(),
    mapping,
    do_atomic_virial=do_atomic_virial,
)
```

该函数用 `mapping` 把 extended atom 域的 force/virial scatter_reduce 回 local atom 域。

这个步骤和 GP 本身不是同一个概念, 但它是 DeePMD extended/ghost atom 语义下最终输出正确的必要收尾。

## 完整调用逻辑

以 `run_4rank_single_gpu_gp.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json` 为例:

```text
run_4rank_single_gpu_gp.sh
  -> 设置 DISABLE_GP_MODE=0
  -> 设置 DP_GP_EXEC_MODE=distributed
  -> 设置 DP_DP_SIZE=2, DP_GP_SIZE=2, DP_PP_SIZE=1, DP_EP_SIZE=1
  -> torch.distributed.run --nproc_per_node=4 tools/run_gp_pt_train.py ...

tools/run_gp_pt_train.py
  -> sys.path 插入 repo root
  -> get_repo_distutils()
  -> distutils.setup(dp=2, gp=2, pp=1, ep=1)
       -> init_process_group
       -> setup_dist_group
            -> DeviceMesh(pp, dp, ep, gp)
            -> _GP_GROUP
            -> _DATA_GROUP
            -> _DP_GP_GROUP / _DP_GP_EP_GROUP
  -> deepmd_main(["--pt", "train", ...])

deepmd/pt/entrypoints/main.py
  -> j_loader(input.json)
  -> normalize config
  -> get_trainer(...)
       -> DpLoaderSet(... seed=[data_rank, seed])
       -> Trainer(...)

deepmd/pt/utils/dataloader.py
  -> _get_data_parallel_rank_world()
  -> DistributedSampler(num_replicas=data_world_size, rank=data_rank)
  -> 同一个 GP group 内 ranks 拿同一份 batch

Trainer.__init__
  -> get_model_for_wrapper(...)
  -> ModelWrapper(...)
  -> 如果 data_world_size > 1:
       DDP(wrapper, process_group=DATA_GROUP)
     else:
       synchronize_parameters over GP group

Trainer.run
  -> get_data()
  -> wrapper(**input_dict, label=label_dict)
  -> EnergyStdLoss.forward
  -> model(**input_dict)

make_model.CM.forward_common
  -> extend_input_and_build_neighbor_list
  -> forward_common_lower

forward_common_lower
  -> atomic_model.forward_common_atomic
       -> DPAtomicModel.forward_atomic
            -> DPA3.forward
                 -> RepFlows.forward
                      -> env matrix / dynamic edge graph
                      -> build gp_partitions
                      -> for each RepFlowLayer:
                           -> pick local partition
                           -> slice local edge/angle/h2/sw/a_sw
                           -> ll.forward(full_node, local_graph)
                           -> keep node_ebd_local
                           -> if not last layer:
                                gather_node_tensor(node_ebd_local)
                              else:
                                return node_ebd_parts
                 -> DPA3 returns GP dict
            -> fitting_net on local descriptor partitions
       -> BaseAtomicModel apply_out_stat/mask per partition
  -> make_model GP merge:
       -> fit_output_to_model_output per local part
       -> gather atom outputs with no-sum-grad
       -> reduce energy/force/virial over GP group
  -> communicate_extended_output(mapping)
  -> output_type_cast

loss.backward
  -> DDP hooks average gradients over DATA_GROUP if DP>1
  -> gather_node_tensor backward reduces node grads over GP group
  -> _maybe_sync_graph_parallel_gradients
       -> allreduce all params over GP group, average=False
  -> optimizer.step
```

## 为什么中间层需要 gather 完整 node

当前每个 partition 只拥有:

```text
中心 node 属于本 partition 的 edge/angle
```

但这些 edge/angle 的 neighbor node 可能属于其他 partition。

所以 layer 计算时传入完整 `node_ebd`, 而 edge/angle 是局部的:

```text
full node_ebd + local edge_index/angle_index
```

当前层结束后, 每个 rank 只保留自己中心 node 的更新结果。

下一层仍然可能需要访问任意 neighbor node 的 embedding, 所以非最后一层必须:

```text
all-gather 所有 node shard -> 完整 node_ebd
```

这就是:

```python
graph_parallel.gather_node_tensor(node_ebd_local, dim=0)
```

的原因。

## 为什么最后一层不 gather

最后一层后, descriptor 已经可以交给 fitting net。fitting net 是 atom-wise 的, 不再需要跨 node 的 message passing。

因此最后一层保留分片:

```text
node_ebd_parts
```

然后每个 GP rank 只对自己负责的 atom descriptor 跑 fitting。

这样可以避免最后一次不必要的 full node gather, 并让原子级输出保持 shard 形式, 到模型输出阶段再统一 gather/reduce。

## GP 和 DP 梯度语义

DP 和 GP 的同步语义不同:

```text
DP: 不同样本, 梯度要平均。
GP: 同一样本的不同 shard, 梯度要求和。
```

代码中对应:

- DDP in DATA_GROUP: 数据并行平均。
- `gather_node_tensor` backward: 中间 node embedding 梯度在 GP group 求和。
- `_maybe_sync_graph_parallel_gradients`: 参数梯度在 GP group 求和, `average=False`。

这也是为什么 DATA group 不能包含 GP 维。如果 DDP 直接用 world group, GP ranks 会被当成不同样本做平均, 语义会错。

## single-process-loop 的价值

`DP_GP_EXEC_MODE=single-process-loop` 不启动真实 GP group。

它在一个进程里:

```text
partition 0 -> ll.forward
partition 1 -> ll.forward
...
```

非最后层用 `torch.cat` 替代 distributed all-gather。

这个模式主要用于:

- 验证 partition 逻辑。
- 对比 GP 与全图输出。
- 排除 NCCL/DeviceMesh/多进程通信问题。

如果 single-process-loop 都不等价, 问题通常在切图、edge/angle remap 或输出合并。

如果 loop 等价但 distributed 不等价, 问题通常在 group、gather/reduce 或梯度同步。

## 当前实现的关键假设

1. GP 只在 DPA3/RepFlow dynamic selection 路径中真正生效。
2. 每个 GP group 内的 ranks 必须拿同一个 datapoint。
3. 节点分区是连续 node id 区间。
4. edge/angle 都按中心 node 归属切分。
5. RepFlow layer 内局部子图仍可访问完整 node embedding。
6. 中间层需要 gather 完整 node embedding。
7. 最后一层保留 node descriptor shard。
8. fitting net 可以按 atom shard 独立执行。
9. 原子级输出是 gather 语义。
10. frame/global 输出是 reduce/sum 语义。

## 当前限制和风险

1. 当前不是最优图划分。连续 node id 切分可能造成大量跨分区 neighbor 依赖。
2. 中间层每层都 all-gather 完整 node embedding, 通信和显存仍较重。
3. 每个 layer 仍传完整 node_ebd 给局部图计算, 所以不是完全意义上的 node memory sharding。
4. GP 输出合并路径较复杂, force/virial 需要特别关注 shape 和 reduce 语义。
5. 主路径中仍能看到一些调试输出和历史调试代码, 正式长跑前需要检查是否有 `print`/`ipdb`/debug env 影响。
6. 普通 `dp --pt train` 不一定初始化仓库 distutils GP; 推荐使用 `tools/run_gp_pt_train.py` 或对应 shell 脚本。
7. GP 与 EP/FSDP 的组合在 `distutils.py` 中有接口, 但实际稳定性要按具体分支和配置验证。

## 和 deepmd-kit-moe 当前 flat+MoE 的关系

这套 `deepmd-kit-arch` 图并行与当前 `deepmd-kit-moe` 的 flat mixed-batch path 关注点不同:

```text
deepmd-kit-arch GP:
  同一个 dense datapoint 内, 按 node partition 拆 RepFlow 图计算。

deepmd-kit-moe flat mixed batch:
  把 mixed-nloc LMDB batch flatten 成 flat graph, 使模型能 forward mixed batch。

MoE EP:
  按 router 把 node/edge/angle token 发到 expert 所在 GPU。
```

也就是说:

- GP 切的是同一张图的 node owner 区间。
- flat mixed batch 解决的是不同 nloc frame 的数据表达。
- MoE EP 切的是 expert 参数和 token dispatch。

如果未来要把 GP 思路合入 MoE/flat, 要先统一三个索引体系:

```text
flat atom index / batch-ptr
RepFlow edge_index / angle_index
GP local_start/local_end partition
MoE expert routing global_eid/local_eid
```

## 最短记忆版

```text
run script
  -> tools/run_gp_pt_train.py
  -> distutils.setup(DeviceMesh pp,dp,ep,gp)
  -> graph_parallel.py exposes gp_rank/gp_world/gather/reduce
  -> DataLoader uses DATA_GROUP, so GP ranks share same datapoint
  -> DDP uses DATA_GROUP, not world group
  -> RepFlows builds dynamic edge/angle graph
  -> nodes split by continuous ranges
  -> edges/angles assigned by owner node
  -> each layer computes local graph
  -> non-last layers gather node shards back to full node embedding
  -> last layer keeps descriptor shards
  -> fitting per shard
  -> atom outputs gather, frame outputs reduce
  -> backward: DP average + GP sum
```
