# 从两卡 GP 到四卡 DP×GP 的改动总结

更新时间：2026-03-25

## 1. 这份文档的范围

这份文档只总结这一轮从：

- 两卡纯图并行：`DP=1, GP=2`

扩展到：

- 四卡数据并行 × 图并行：`DP=2, GP=2`

过程中涉及的实际代码改动、文件位置、作用分工和验证结果。

这里不重复整个项目的总历史，而是只回答一个问题：

> 为了把现有两卡 GP 运行链路扩展成四卡 DP×GP，我到底改了哪些地方，它们分别负责什么。

---

## 2. 目标语义

这一轮最终确认的目标语义是：

1. DataLoader 每次送进模型的是一个 datapoint，而不是单个图；
2. 一个 datapoint 来自同一个 system，因此 batch 内多个 frame 具有相同原子数；
3. DP 是外层样本级复制：不同 DP replica 处理不同 datapoint；
4. GP 是内层 datapoint 级切分：同一个 datapoint 在本 replica 内沿展平后的节点维切成两份；
5. backward 时先做 GP 内梯度同步，再做 DP 间梯度平均。

对应的两种运行模式是：

- 两卡纯 GP：`DP=1, GP=2`
- 四卡 DP×GP：`DP=2, GP=2`

---

## 3. 改动总览

这次改动分成两层：

### 3.1 两卡 GP 时代已经打下的基础

这些改动最初是为两卡 `DP=1, GP=2` 打通的，但四卡 `DP=2, GP=2` 仍然直接依赖它们：

- `tools/run_gp_pt_train.py`
- `deepmd/utils/local_distutils.py`
- `deepmd/pt/utils/dataloader.py`
- `deepmd/pt/entrypoints/main.py`
- `deepmd/pt/train/training.py`
- `distutils.py`
- `graph_parallel.py`
- `deepmd/pt/model/descriptor/repflows.py`

### 3.2 为四卡 DP×GP 额外补上的改动

这是这轮新增或强化的部分：

- `run_4rank_single_gpu_gp.sh`
- `run_2rank_gp.sh`
- `distutils.py` 中 4D `DeviceMesh` 复合 group 回退逻辑
- `distutils.py` 中 2×2 拓扑日志
- `deepmd/pt/train/training.py` 中 2×2 batch 观测日志
- `deepmd/pt/model/descriptor/repflows.py` 中 datapoint 级 GP 分片观测日志

---

## 4. 启动层改动

### 4.1 `tools/run_gp_pt_train.py`

作用：

- 作为 GP / DP×GP 的统一 Python 启动包装层；
- 从环境变量读取并行拓扑；
- 在进入 DeePMD PyTorch 主入口前调用仓库版 `distutils.setup(...)`。

关键改动：

1. 将 repo 根目录插入 `sys.path`；
2. 通过 `get_repo_distutils()` 强制加载仓库自己的 `distutils.py`；
3. 新增 `_maybe_setup_distributed_gp()`：
   - 读取 `DP_DP_SIZE / DP_GP_SIZE / DP_PP_SIZE / DP_EP_SIZE`
   - 校验 `PP * DP * EP * GP == WORLD_SIZE`
   - 调用 `gp_distutils.setup(...)`
4. 保留 `debugpy` 的按 rank 端口映射能力。

意义：

- 两卡纯 GP 和四卡 DP×GP 最终都通过这个入口完成分布式初始化；
- 它把 shell 环境变量和 Python 内部的 DeviceMesh 初始化桥接起来。

### 4.2 `deepmd/utils/local_distutils.py`

作用：

- 解决仓库根目录 `distutils.py` 与 Python 标准库 `distutils` 同名冲突。

关键改动：

- 使用 `importlib.util.spec_from_file_location(...)` 显式加载仓库版 `distutils.py`。

意义：

- 否则分布式初始化会导入错模块，导致 GP / DP×GP 启动链路直接失效。

### 4.3 `run_2rank_gp.sh`

当前定位：

- 保留为两卡纯 GP 入口。

默认拓扑：

- `DP=1`
- `GP=2`
- `PP=1`
- `EP=1`
- `CUDA_VISIBLE_DEVICES=0,1`

关键改动：

1. 明确脚本职责是两卡纯 GP；
2. 默认 `DP_DP_SIZE=1`、`DP_GP_SIZE=2`；
3. 默认只使用 0、1 号 GPU；
4. 保留 `CONDA_ENV_PREFIX=/aisi-nas/liwentao/miniconda/deepmd_gp`；
5. 默认关闭 2×2 bring-up 日志：`DP_DEBUG_2X2=0`。

意义：

- 保留原本两卡 `DP=1, GP=2` 的可复用入口，不让四卡脚本覆盖掉旧工作流。

### 4.4 `run_4rank_single_gpu_gp.sh`

当前定位：

- 独立承载四卡 `DP=2, GP=2` 入口。

默认拓扑：

- `DP=2`
- `GP=2`
- `PP=1`
- `EP=1`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`

关键改动：

1. 不再只是转发到两卡脚本，而是独立维护四卡拓扑；
2. 默认 `NPROC_PER_NODE=4`；
3. 默认打开 bring-up 日志：`DP_DEBUG_2X2=1`；
4. 在启动前打印：
   - log 目录
   - 并行拓扑
   - GPU 可见性
   - conda 环境位置。

意义：

- 两卡与四卡入口完全解耦；
- 脚本名与实际默认行为一致，不再混淆。

---

## 5. 分布式初始化层改动

### 5.1 `distutils.py`

这是本轮最核心的分布式基础设施文件。

#### 已有的两卡 GP 基础能力

1. `DeviceMesh` 导入兼容：
   - 优先 `torch.distributed.tensor.DeviceMesh`
   - 回退到 `torch.distributed.device_mesh.DeviceMesh`
2. `janus.core.distutils` 可选导入；
3. `setup(...)` 中根据 `LOCAL_RANK / RANK / WORLD_SIZE` 初始化 torch distributed；
4. 构建 `PP / DP / EP / GP` 各维 process group；
5. 提供：
   - `get_data_rank()`
   - `get_data_world_size()`
   - `get_gp_rank()`
   - `get_gp_world_size()`
   - `get_data_group()`
   - `get_gp_group()`
6. 提供参数同步与梯度 allreduce 工具。

#### 为四卡 DP×GP 新增的关键改动

1. 新增 `_manual_mesh_group(mesh, vary_dim_names)`：
   - 当 PyTorch 2.3 的 4D `DeviceMesh` 不支持 `mesh[("dp", "gp")]` 这类 tuple flatten 时，
   - 手动根据 mesh 维度构造复合 subgroup。

2. 在 `setup_dist_group(...)` 中对以下 group 增加手动回退：
   - `DP×GP`
   - `DP×EP×GP`
   - `DP×EP`

3. 新增 `_safe_group_info(...)`：
   - 安全读取 group world size 和成员 rank 列表。

4. 新增 2×2 bring-up 拓扑日志：
   - `global rank`
   - `data rank`
   - `gp rank`
   - `DP / DATA / GP / PP / EP / DPxGP` 组成员。

意义：

- 真正解决了 `DP=2, GP=2` 在 PyTorch 2.3 上无法完成 `DeviceMesh` 复合 group 构造的问题；
- 也是四卡 2×2 能启动并进入训练的关键修复点。

---

## 6. 数据加载与 trainer 输入层改动

### 6.1 `deepmd/pt/utils/dataloader.py`

作用：

- 让数据切分按 data-parallel 语义进行，而不是按 global world size 进行。

关键改动：

1. 新增 `_get_data_parallel_rank_world()`：
   - 优先从 `gp_distutils.get_data_rank()` / `get_data_world_size()` 取值；
   - 没有时再回退到原始 `dist.get_rank()` / `dist.get_world_size()`。

2. `DistributedSampler(...)` 使用：
   - `num_replicas=data_world_size`
   - `rank=data_rank`

而不是使用全局 world size。

意义：

- 在四卡 `DP=2, GP=2` 下，rank 0 和 1 会拿到同一个 datapoint，rank 2 和 3 会拿到另一个 datapoint；
- 这正是“DP 是宏观样本复制，GP 是 datapoint 内部细粒度切分”的前提。

### 6.2 `deepmd/pt/entrypoints/main.py`

作用：

- 准备 trainer 输入时区分 global rank 与 data rank。

关键改动：

1. 新增 `_get_data_parallel_rank()`；
2. `prepare_trainer_input_single(...)` 扩展为接收：
   - `global_rank`
   - `data_rank`
3. `stat_file` 这类全局行为只由 `global_rank == 0` 负责；
4. `rank_seed` 改成基于 `data_rank` 混入。

意义：

- 防止纯 GP 或 DP×GP 场景里，随机种子与数据分片错误地按全局 rank 演化。

---

## 7. 训练同步层改动

### 7.1 `deepmd/pt/train/training.py`

作用：

- 决定 DDP 包裹范围，以及 GP 梯度同步和参数同步策略。

#### 原有两卡 GP 基础改动

1. 新增 `_get_data_parallel_world_size()` 和 `_get_data_parallel_group()`；
2. 只有 `data_world_size > 1` 时才用 DDP；
3. 纯 GP 时不使用 DDP，而是只做 GP 参数同步；
4. 在 `loss.backward()` 后调用 `_maybe_sync_graph_parallel_gradients(...)`，用 GP group 做额外 allreduce。

这部分确保了：

- 纯 GP 时梯度语义正确；
- 四卡 DP×GP 时依然保持“先 GP 内同步，再 DP 间平均”的两层结构。

#### 为四卡 DP×GP 新增的可观测性改动

1. 新增 `_debug_2x2_enabled()`；
2. 新增 `_get_data_parallel_rank()` 和 `_get_graph_parallel_rank()` 辅助函数；
3. 在训练 step 中增加一次性日志：
   - `rank`
   - `data_rank`
   - `gp_rank`
   - `sid`
   - `fid`
   - `coord_shape`
   - `nframes`
   - `nloc`

意义：

- 这是验证四卡 2×2 语义是否正确的最直接证据；
- 日志已经证明：
  - rank 0/1 看到同一个 datapoint；
  - rank 2/3 看到另一个 datapoint。

---

## 8. 模型内 GP 切分层改动

### 8.1 `deepmd/pt/model/descriptor/repflows.py`

这是图并行切分真正发生的地方。

#### 原有两卡 GP 基础改动

1. 根据环境变量判断 GP 执行模式：
   - distributed
   - single-process-loop
2. 在 `use_dynamic_sel=True` 时构造 `gp_partitions`；
3. `gp_num_nodes = node_ebd.shape[0]`，也就是对 datapoint 展平后的节点总数做切分；
4. 通过 `_balanced_partition_sizes(...)` 和 `_build_partition_offsets(...)` 生成：
   - `local_start`
   - `local_end`
   - `local_size`
5. 边和角都按 owner node 所属区间切分，并做 edge id remap；
6. distributed 模式下，每个 GP rank 只执行自己的 partition；
7. 非最后层对 `node_ebd_local` 做跨 rank gather；
8. 最后层把 `gp_partitions` 和 `node_ebd_parts` 等结构化地返回上层。

这套逻辑本身最初就是两卡纯 GP 的基础。

#### 为四卡 DP×GP 新增的改动

1. 新增 logger：`log = logging.getLogger(__name__)`；
2. 新增 `_GP_PARTITION_DEBUG_PRINTED`，避免日志过量重复；
3. 在 `DP_DEBUG_2X2=1` 且 distributed GP 时，打印一次性分片日志：
   - `rank`
   - `gp_rank`
   - `gp_num_nodes`
   - `local_start`
   - `local_end`
   - `local_size`
   - `edge_count`
   - `angle_count`

意义：

- 直接验证 GP 不是按单图切，而是按 datapoint 的展平节点维切；
- 在真实四卡日志中已经观察到：
  - `coord_shape=(2,108,3)` 对应 `gp_num_nodes=216 -> 108/108`
  - `coord_shape=(2,104,3)` 对应 `gp_num_nodes=208 -> 104/104`

这正符合我们最终确认的 datapoint 级切分语义。

---

## 9. 这轮没有重写、但仍被依赖的地方

以下文件在这轮 4 卡 bring-up 中没有做大改，但它们仍是两卡 GP 演进到四卡 DP×GP 的基础：

- `graph_parallel.py`
  - 继续承担 `get_gp_rank()`、`get_gp_world_size()`、`gather_node_tensor()` 等抽象；

- `deepmd/pt/model/descriptor/dpa3.py`
  - 继续透传 GP 分片结构；

- `deepmd/pt/model/atomic_model/dp_atomic_model.py`
  - 继续对每个 partition 单独做 fitting；

- `deepmd/pt/model/model/make_model.py`
  - 继续对 partition 结果做模型级合并；

- `deepmd/pt/model/model/transform_output.py`
  - 继续承担最终 output transform。

换句话说，这次四卡 DP×GP 扩展主要是把外层拓扑与数据路由打通，而不是重写模型公式或 output transform 主逻辑。

---

## 10. 实际验证结果

这轮已经完成的验证包括：

### 10.1 两卡 `DP=1, GP=2`

- 用户已确认两卡纯 GP 梯度没有问题；
- 两卡脚本已保留为独立入口：`run_2rank_gp.sh`。

### 10.2 四卡 `DP=2, GP=2`

在用户指定环境 `/aisi-nas/liwentao/miniconda/deepmd_gp` 中，已真实完成以下验证：

1. 4 张 GPU 可见；
2. 四卡 launcher 能启动；
3. 2×2 的 group 关系正确打印；
4. 同一个 DP replica 内两个 GP rank 拿到同一个 datapoint；
5. GP 分片按 datapoint 的 `nf * nloc` 节点维二分；
6. 训练已进入实际 batch 循环，并持续运行数百步。

对应日志文件：

- `logs/torchrun/dp2_gp2_minimal_validation.log`

其中可以直接看到：

- rank 0/1: 同一个 `sid/fid`、同一个 `coord_shape`
- rank 2/3: 同一个 `sid/fid`、同一个 `coord_shape`
- 例如：
  - `(2,108,3) -> gp_num_nodes=216 -> 108/108`
  - `(2,104,3) -> gp_num_nodes=208 -> 104/104`

---

## 11. 当前最终脚本分工

### 两卡纯 GP

脚本：`run_2rank_gp.sh`

默认行为：

- `DP=1`
- `GP=2`
- `CUDA_VISIBLE_DEVICES=0,1`

示例命令：

```bash
CONDA_ENV_PREFIX=/aisi-nas/liwentao/miniconda/deepmd_gp \
bash ./run_2rank_gp.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json
```

### 四卡 DP×GP

脚本：`run_4rank_single_gpu_gp.sh`

默认行为：

- `DP=2`
- `GP=2`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`

示例命令：

```bash
CONDA_ENV_PREFIX=/aisi-nas/liwentao/miniconda/deepmd_gp \
bash ./run_4rank_single_gpu_gp.sh train --skip-neighbor-stat test_mptraj/input_dpa3.json
```

---

## 12. 一句话总结

这一轮从两卡 GP 到四卡 DP×GP 的核心，不是重写模型，而是：

1. 保留两卡纯 GP 作为稳定基线；
2. 在同一套 `DeviceMesh` / `distutils` 基础设施上，把外层 data-parallel 复制维补出来；
3. 让 dataloader 按 data group 送不同 datapoint；
4. 让 `repflows.py` 继续在每个 datapoint 内沿展平节点维做 GP 二分；
5. 用真实四卡训练日志验证这一整套语义已经成立。
