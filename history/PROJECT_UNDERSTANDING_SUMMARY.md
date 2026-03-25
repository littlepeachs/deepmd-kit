# 项目整体理解总结

更新时间：2026-03-10

## 1. 项目定位

这个仓库基于 DeePMD-kit，核心目标仍然是深度势能模型训练与推理，但当前分支的重点不是原生主线功能，而是围绕 **PyTorch 后端的图并行（Graph Parallel, GP）改造** 做实验与验证。

当前工作的主线可以概括为：

1. 保留原始单图、单进程、非图并行训练路径；
2. 在 PyTorch 训练路径中插入一套 GP 执行逻辑；
3. 支持两种 GP 运行模式：
   - **分布式双卡图并行**：一个完整图拆成两个子图，分别落到两张 GPU；
   - **单进程串行图并行模拟**：在一张卡/一个进程里，用 `for` 循环依次执行两个子图，模拟 GP 行为；
4. 最终希望将 **“不做图并行的完整图结果”** 与 **“做图划分后的重组结果”** 做严格对比，验证数值一致性。

因此，这个分支的本质不是简单“能跑起来”，而是要验证：

- 图的划分是否正确；
- 子图计算后如何重新拼回原图语义；
- 分布式 GP 与单进程 loop GP 是否与 full-graph 前向一致。

---

## 2. 当前理解下的整体执行链路

### 2.1 基线运行路径

基线脚本是仓库根目录下的 [run_original.sh](../run_original.sh)。

它的作用是关闭 GP，直接走原始完整图训练路径。其核心行为是：

- 设置 `DISABLE_GP_MODE=1`；
- 调用 `dp --pt train --skip-neighbor-stat ./test_mptraj/input_dpa3.json`。

这条路径代表“未做图并行”的参考结果。

### 2.2 PyTorch 训练入口

训练命令最终会进入 DeePMD 的 PyTorch 入口链路：

- [deepmd/main.py](../deepmd/main.py)
- [deepmd/pt/entrypoints/main.py](../deepmd/pt/entrypoints/main.py)
- [deepmd/pt/train/training.py](../deepmd/pt/train/training.py)

模型前向主链继续向下进入：

- [deepmd/pt/model/model/ener_model.py](../deepmd/pt/model/model/ener_model.py)
- [deepmd/pt/model/model/make_model.py](../deepmd/pt/model/model/make_model.py)
- [deepmd/pt/model/atomic_model/base_atomic_model.py](../deepmd/pt/model/atomic_model/base_atomic_model.py)
- [deepmd/pt/model/atomic_model/dp_atomic_model.py](../deepmd/pt/model/atomic_model/dp_atomic_model.py)
- [deepmd/pt/model/descriptor/dpa3.py](../deepmd/pt/model/descriptor/dpa3.py)
- [deepmd/pt/model/descriptor/repflows.py](../deepmd/pt/model/descriptor/repflows.py)
- [deepmd/pt/model/model/transform_output.py](../deepmd/pt/model/model/transform_output.py)

这条链路就是当前图并行改造要打通的主路径。

---

## 3. GP 改造的核心目标与方案

### 3.1 目标

用户的明确目标是：

- 把“一个完整图”拆成“两个子图”；
- 在 **2 张 GPU 上并行执行**；
- 或者在 **单进程里串行执行两个子图**；
- 在模型输出端将子图结果重新组合；
- 与原始完整图执行结果进行比较。

### 3.2 当前采用的图划分方式

在当前实现中，`repflows.py` 中的 GP 划分不是复杂的图优化切分，而是更接近 **按节点区间的均衡切分**：

- 先根据 GP world size 计算每个 partition 的节点数；
- 再得到每个子图的 `local_start` 和 `local_end`；
- 通过 mask 选出属于该区间的边、角等结构；
- 对每个 partition 构造局部输入并执行每层 `RepformerLayer`。

也就是说，当前是“便于验证正确性”的切分策略，而不是“最优通信代价”的切分策略。

---

## 4. 设备拓扑与 DeviceMesh 的作用

### 4.1 拓扑设计

当前 GP 运行的默认目标拓扑是：

- `PP = 1`
- `DP = 1`
- `EP = 1`
- `GP = 2`

也就是一个 **纯图并行** 的 2 卡拓扑，不做数据并行。

### 4.2 DeviceMesh 在这里的意义

[distutils.py](../distutils.py) 是整个并行拓扑的中心。这里会基于 `(pp, dp, ep, gp)` 创建 `DeviceMesh`，并从中抽取不同维度的通信组。

在当前目标配置下，最重要的是 GP 维：

- 每个 rank 对应一个图分片；
- 后续的 gather/reduce 会围绕 GP group 进行；
- 纯 GP 模式下，很多传统 DP 语义不能直接照搬，所以做了一些回退和兼容处理。

### 4.3 兼容性问题与已解决点

为了让当前环境能工作，已经处理过以下问题：

1. **PyTorch 2.3 的 `DeviceMesh` 导入兼容性**；
2. **`janus` 缺失时的可选导入问题**；
3. **纯 GP (`DP=1, EP=1, GP=2`) 下 group tuple 展平失败**，因此增加了纯 GP 回退逻辑。

这说明当前分支的并行层已经不再是“理论设计”，而是针对实际环境调通过一轮。

---

## 5. 当前运行模式

### 5.1 分布式双卡 GP

启动脚本是 [run_4rank_single_gpu_gp.sh](../run_4rank_single_gpu_gp.sh)。

虽然脚本名保留了旧命名，但当前实际目标是：

- 使用用户指定环境 `/aisi-nas/liwentao/miniconda/deepmd_gp`；
- 默认按 `GP=2` 启动；
- 通过 `python -m torch.distributed.run` 进入 GP 训练入口 [tools/run_gp_pt_train.py](../tools/run_gp_pt_train.py)。

[tools/run_gp_pt_train.py](../tools/run_gp_pt_train.py) 的职责是：

- 补齐 repo 根路径到 `sys.path`；
- 强制加载仓库自己的 `distutils.py`，避免导入到 Python 标准库 `distutils`；
- 根据环境变量读取 `PP/DP/EP/GP`；
- 做拓扑合法性校验；
- 初始化 GP 分布式环境；
- 最后转入 `deepmd_main(["--pt", ...])`。

### 5.2 单进程串行 GP loop

启动脚本是 [run_single_process_gp_loop.sh](../run_single_process_gp_loop.sh)。

它的目标不是做真正的多卡分布式，而是：

- 在单进程内模拟 `GP world size = 2`；
- 在 `repflows.py` 内部用 `for partition in gp_partitions` 依次执行两个子图；
- 验证即使不依赖分布式通信，纯粹串行切图后再组合，逻辑是否正确。

这个模式对定位模型层面的拼接错误非常重要，因为它能把“通信问题”和“算法问题”分离开。

---

## 6. GP 改造涉及的关键文件及职责

### 6.1 启动与并行初始化

- [run_4rank_single_gpu_gp.sh](../run_4rank_single_gpu_gp.sh)
  - 分布式 GP 启动脚本。
- [run_single_process_gp_loop.sh](../run_single_process_gp_loop.sh)
  - 单进程串行 GP 模拟脚本。
- [tools/run_gp_pt_train.py](../tools/run_gp_pt_train.py)
  - GP 入口包装层。
- [distutils.py](../distutils.py)
  - `DeviceMesh`、group 初始化、collective、日志等。
- [graph_parallel.py](../graph_parallel.py)
  - GP 工具层，提供 rank/world size 和 gather/reduce 抽象。
- [deepmd/utils/local_distutils.py](../deepmd/utils/local_distutils.py)
  - 强制加载仓库本地 `distutils.py`，避免命名冲突。

### 6.2 数据与训练阶段

- [deepmd/pt/utils/dataloader.py](../deepmd/pt/utils/dataloader.py)
  - 让 DataLoader 的 sampler 按 data-parallel 语义而不是 raw world-size 工作。
- [deepmd/pt/entrypoints/main.py](../deepmd/pt/entrypoints/main.py)
  - 区分 `global_rank` 与 `data_rank`；
  - `stat_file` 这类全局行为由 global rank 控制；
  - 数据随机种子按 data rank 控制。
- [deepmd/pt/train/training.py](../deepmd/pt/train/training.py)
  - 只有真正存在 DP 时才使用 DDP；
  - 纯 GP 时改为参数同步而不是 DDP 包装。

### 6.3 模型主路径

- [deepmd/pt/model/descriptor/repflows.py](../deepmd/pt/model/descriptor/repflows.py)
  - 图划分与每层子图执行的核心位置。
- [deepmd/pt/model/descriptor/dpa3.py](../deepmd/pt/model/descriptor/dpa3.py)
  - 接住 `repflows` 的 GP 输出，并继续向上层传递。
- [deepmd/pt/model/atomic_model/dp_atomic_model.py](../deepmd/pt/model/atomic_model/dp_atomic_model.py)
  - 对每个 partition 的 descriptor 结果分别做 fitting。
- [deepmd/pt/model/atomic_model/base_atomic_model.py](../deepmd/pt/model/atomic_model/base_atomic_model.py)
  - 对各 partition 输出做统计和 mask。
- [deepmd/pt/model/model/make_model.py](../deepmd/pt/model/model/make_model.py)
  - 将 per-part fitting 输出整理为模型级输出。
- [deepmd/pt/model/model/transform_output.py](../deepmd/pt/model/model/transform_output.py)
  - 做导数、力、virial 等转换，并把 extended 输出映射回 local atom。

---

## 7. 当前最关键的模型层逻辑

### 7.1 `repflows.py`：图划分与逐层执行

这是当前最核心的文件。

它承担的职责包括：

1. 判断是否启用 GP；
2. 构造 `gp_partitions`；
3. 在分布式模式下，让每个 `gp_rank` 只执行自己的 partition；
4. 在单进程 loop 模式下，依次执行所有 partition；
5. 在非最后层时，将子图结果拼接/gather，供下一层继续使用；
6. 在最后层，将局部结果打包成结构化字典向上传递。

当前约定的上行数据通常包括：

- `gp_partitions`
- `local_partition_indices`
- `node_ebd_parts`
- `rot_mat_parts`
- `h2_parts`
- `sw_parts`

这说明在最后一层后，不再试图立刻恢复成传统单张量格式，而是把“分片结构”一直保留到上层。

### 7.2 `dpa3.py`：保留 GP 结构

`dpa3.py` 当前并不是把 GP 输出拍平，而是继续保留分片结构，必要时再做类型嵌入拼接。

因此它更像一个“结构透传 + 局部增强”的中间层。

### 7.3 `dp_atomic_model.py`：每个 partition 分开拟合

进入 fitting 阶段后，不是先合并再拟合，而是：

- 每个 partition 拿自己的 `node_ebd_part`、`atype_part`、`rot_mat_part`、`h2_part` 等；
- 单独调用 `self.fitting_net(...)`；
- 把结果存回 `fit_ret_parts`。

这是很自然的 GP 设计：descriptor 是分片的，fitting 也跟着分片。

### 7.4 `make_model.py` 与 `transform_output.py`：最终拼接与映射

真正最难的是这里。

原因在于：

- 能量、原子量、力、virial 等量的张量语义不一样；
- 有的是节点级，有的是 frame 级；
- 尤其 `force`、`virial` 这类导数量，需要从 extended 坐标关系映射回本地 atom 索引。

这也是目前还没有完全闭环验证成功的地方。

---

## 8. 已经解决过的工程问题

### 8.1 指定 conda 环境

用户明确要求使用：

- `/aisi-nas/liwentao/miniconda/deepmd_gp`

目前 GP 相关脚本都已切到这个环境启动。

### 8.2 `torchrun` 与入口问题

原先的启动方式不适配当前环境，已经改成更稳妥的：

- `python -m torch.distributed.run`

### 8.3 仓库内 `distutils.py` 与 Python 标准库同名冲突

这是一个关键问题。

因为直接 `import distutils` 很容易导入到 Python 标准库，而不是仓库根目录下的 [distutils.py](../distutils.py)。

已经通过新增 [deepmd/utils/local_distutils.py](../deepmd/utils/local_distutils.py) 解决，后续 GP 入口和 helper 都改为显式加载仓库版本。

### 8.4 日志格式中的 `rank` 报错

之前执行 GP 训练时出现过：

- `Formatting field not found in record: 'rank'`

原因是日志 formatter 依赖 `%(rank)s`，但部分 log record 并没有被注入 `rank` 字段。

已经在 [distutils.py](../distutils.py) 中把 `RankFilter` 同时挂到 handler 和 root logger 上，帮助信息路径验证后，这个报错已经消失。

---

## 9. 当前尚未完全解决的问题

### 9.1 输出重组仍然是核心难点

当前最主要的未完成问题集中在 [deepmd/pt/model/model/transform_output.py](../deepmd/pt/model/model/transform_output.py) 的 `communicate_extended_output()` 一带。

问题本质是：

- `local_start/local_end` 更接近“owner 节点区间”的概念；
- `mapping`、`extended_coord`、scatter 索引则处在另一套索引语义里；
- 当子图导数量回写到完整图局部原子时，很容易出现 shape 不匹配或索引空间不一致。

### 9.2 已经暴露过的具体症状

之前已经见到过两类典型错误：

1. `torch.scatter_reduce` 阶段 index/self shape mismatch；
2. 后续 loss 计算中，预测力和标签力维度不一致，例如 `108` 对 `216`。

这说明：

- 前向分片本身已经深入到模型后段；
- 但最终“映射回统一输出”的语义尚未完全对齐。

### 9.3 当前 patch 的含义

[transform_output.py](../deepmd/pt/model/model/transform_output.py) 已经做过一轮修补：

- 区分 `mapping_r` 和 `mapping_c`；
- 输出 leading dims 根据 `mapping` 推断；
- `nloc` 从 `mapping.max() + 1` 推断。

这些补丁说明方向是对的：当前问题不是简单张量拼接，而是“索引空间恢复”问题。

但是否已经完全修好，还没有得到最终验证。

---

## 10. 当前对 `repflows.py` 调试状态的理解

[deepmd/pt/model/descriptor/repflows.py](../deepmd/pt/model/descriptor/repflows.py) 目前仍然是调试高频区。

文件中曾经出现或仍然出现：

- `print(...)`
- `pdb/ipdb.set_trace()`
- 按 `RANK` 分支打印局部张量

这说明用户当前主要在确认：

- 每个 rank 究竟拿到了哪个 partition；
- 最后一层 `node_ebd` 到底是全局拼接后的结果，还是本 rank 的局部结果；
- 为什么某些打印看起来像“rank 0 和 rank 1 都在 rank 0 打印”。

对这个现象的当前理解是：

1. 如果代码写成 `if int(os.environ.get("RANK", "0")) == 0` 与 `if int(os.environ.get("RANK", "1")) == 1`，那么在 **没有 `RANK` 环境变量的单进程模式** 下，两边都可能为真；
2. 如果第二个条件其实误写成 `== 0`，那么在 **分布式模式** 下两边都会在 rank 0 打印；
3. 即使条件没写错，在分布式 GP 最后阶段，`node_ebd[0]` 也可能只是“本 rank 的局部 partition 张量”，而不是 gather 后的全局张量，因此你在 rank 0 上打印 `node_ebd[0][-6:, :6]`，看到的仍然只是 rank 0 本地张量尾部，不是 rank 1 的数据。

也就是说，这个问题不一定是“打印系统错了”，很可能是：

- 环境变量判断方式本身有歧义；
- 当前变量语义是局部张量，而不是全局张量。

---

## 11. 当前可确认的阶段性成果

截至目前，可以确认以下几点已经完成：

1. 已经完整梳理出基线训练路径与 GP 改造路径；
2. 已经把分布式双卡 GP 启动链路打通到真实模型执行阶段；
3. 已经把单进程串行 GP loop 路径打通到真实模型执行阶段；
4. 已经解决启动层面的环境、导入、日志、DeviceMesh 兼容等问题；
5. 已经形成了较完整的工程文档，对实现流程和模式对比做了说明。

换句话说，目前系统已经跨过了“怎么让它启动”的阶段，进入到了“如何保证图切分后数值语义正确”的阶段。

---

## 12. 当前最合理的后续方向

### 12.1 第一优先级：验证 `repflows` 最后一层输出语义

首先需要回答清楚：

- `node_ebd` 在最后一层返回时是局部还是全局；
- 分布式模式与 single-process-loop 模式下它的结构是否一致；
- `node_ebd_parts` / `local_partition_indices` 与原图节点索引的对应关系是否稳定。

如果这一步不清楚，后续 `transform_output.py` 很难完全修正。

### 12.2 第二优先级：统一 output transform 的索引语义

需要明确区分：

- owner 节点索引；
- extended 原子索引；
- local 原子索引；
- force/virial 使用的 scatter 索引。

当前很多报错都指向这些索引空间混用了。

### 12.3 第三优先级：建立严格的 full-graph vs GP 对照验证

最终应形成固定验证流程：

1. 关闭 GP 跑一遍 full graph；
2. 开 GP 跑 distributed；
3. 开 GP 跑 single-process-loop；
4. 对比 descriptor 中间结果、fitting 输出、最终 energy/force/virial；
5. 逐层定位偏差源头。

---

## 13. 相关说明文档

当前仓库中已经补充了几份说明文档，可作为后续阅读入口：

- [GRAPH_PARALLEL_CURRENT_IMPLEMENTATION.md](../GRAPH_PARALLEL_CURRENT_IMPLEMENTATION.md)
- [GRAPH_PARALLEL_CURRENT_IMPLEMENTATION_BRIEF.md](../GRAPH_PARALLEL_CURRENT_IMPLEMENTATION_BRIEF.md)
- [GRAPH_PARALLEL_MODE_COMPARISON.md](../GRAPH_PARALLEL_MODE_COMPARISON.md)
- [GRAPH_PARALLEL_MODIFICATION_SUMMARY.md](../GRAPH_PARALLEL_MODIFICATION_SUMMARY.md)
- [GRAPH_PARALLEL_SCHEMES.md](../GRAPH_PARALLEL_SCHEMES.md)

这些文档更偏向“实现流程说明”，而本文件更偏向“整个项目当前状态与问题理解的历史总结”。

---

## 14. 总结

如果用一句话概括当前项目状态：

> 这个分支已经把 DeePMD-kit 的 PyTorch 训练路径改造成了一个可进入真实模型计算阶段的图并行实验系统，当前主要矛盾已经从“分布式启动与环境兼容”转移到“图划分后的输出重组与数值一致性验证”。

因此，接下来的重点不再是外层工程脚本，而是模型层面三件事：

1. **确认 `repflows` 每层、尤其最后一层的分片输出语义；**
2. **修正 `transform_output.py` 中各类索引空间的映射关系；**
3. **建立 full-graph / distributed GP / single-process-loop GP 的严格逐层对照。**

只要这三件事完成，这个分支就能从“图并行改造实验版”进入“可验证正确性的图并行实现版”。
