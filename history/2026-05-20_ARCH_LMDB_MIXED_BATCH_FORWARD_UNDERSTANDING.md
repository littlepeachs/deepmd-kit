# deepmd-kit-arch LMDB mixed batch forward 学习记录

日期: 2026-05-20

范围: `/aisi-nas/liwentao/deepmd-kit-moe/deepmd-kit-arch`

对比背景: 当前 `deepmd-kit-moe` 的 `deepmd/pt/train/training.py` 里 LMDB `mixed_batch=True` 仍直接 `NotImplementedError`; `deepmd-kit-arch` 已经实现了一条 flat mixed-nloc 路径。

## 结论概览

`deepmd-kit-arch` 没有用 padding + atom_mask 作为主方案, 而是采用 "flatten atoms + precomputed flat graph"。

核心思路:

1. LMDB 每个 frame 可以有不同 `nloc`。
2. collate 时把所有 frame 的 atom-wise 字段沿 atom 维拼平。
3. 额外生成:
   - `batch`: 每个 atom 属于哪个 frame。
   - `ptr`: frame 的 atom prefix sum。
4. collate 同时根据模型 descriptor 的 cutoff/sel 预构建 flat graph:
   - extended atoms
   - neighbor list
   - angle neighbor list
   - valid masks
   - dynamic graph indices
5. 模型 forward 检测到 `batch` 和 `ptr`, 进入 `forward_common_flat()`。
6. descriptor 走 `forward_flat()`, fitting net 暂时把 flat descriptor scatter 到 dense `[nframes, max_nloc, ...]` 跑原有 fitting, 再 gather 回 flat。
7. energy 用 `index_add_(0, batch, atom_energy)` reduce 到 frame 级。
8. force 直接对 flattened `coord` 求导, 输出 `[total_atoms, 3]`。

## 配置和 DataLoader 入口

配置文件示例:

- `test_mptraj/lmdb_mixed_batch.json`
- `training.training_data.mixed_batch: true`
- `training.validation_data.mixed_batch: true`

训练器中 `get_data_loader()` 对 LMDB 数据分两条路径:

```text
LmdbDataset.mixed_batch == True
  -> RandomSampler / SequentialSampler
  -> DataLoader(batch_size=_data.batch_size)
  -> collate_fn = make_lmdb_mixed_batch_collate(graph_config)

LmdbDataset.mixed_batch == False
  -> SameNlocBatchSampler
  -> DataLoader(batch_sampler=...)
  -> collate_fn = _collate_lmdb_batch
```

`graph_config` 从已构造模型的 descriptor 读取:

```python
model_for_graph = self.model[_task_key] if multi_task else self.model
descriptor = model_for_graph.atomic_model.descriptor
graph_config = {
    "rcut": descriptor.get_rcut(),
    "sel": descriptor.get_sel(),
    "a_rcut": descriptor.repflows.a_rcut,
    "a_sel": descriptor.repflows.a_sel,
    "mixed_types": descriptor.mixed_types(),
}
```

因此 mixed batch 当前要求 descriptor 有 `repflows` 属性, 实际就是 flat-graph capable 的 DPA3/RepFlow 路径。

## LMDB Dataset 和 collate

`deepmd/pt/utils/lmdb_dataset.py` 中有两个关键 collate:

- `_collate_lmdb_mixed_batch(batch)`
- `make_lmdb_mixed_batch_collate(graph_config)`

### `_collate_lmdb_mixed_batch()`

它先读取每个 frame 的 atom 数:

```text
counts = [len(item["atype"]) for item in batch]
ptr = [0, counts[0], counts[0]+counts[1], ...]
batch = repeat_interleave(arange(nframes), counts)
```

返回格式:

- atom-wise key 直接 `torch.cat(tensors, dim=0)`
  - `coord`: `[total_atoms, 3]`
  - `atype`: `[total_atoms]`
  - `force`: `[total_atoms, 3]`
  - `aparam`, `atom_ener`, `atom_pref`, `spin`, `hessian` 等同理
- frame-wise key 保持 batch 维度, 用 torch 默认 collate
  - `energy`: `[nframes, ...]`
  - `box`: `[nframes, 9]`
  - `virial`: `[nframes, 9]`
- `batch`: `[total_atoms]`, atom -> frame
- `ptr`: `[nframes + 1]`
- `sid`: 固定为 shape `[1]` 的 CPU tensor
- `fid`: 保持 list

这里没有把不同 `nloc` padding 到同一个 dense batch。dense 只在 fitting net 内部临时出现。

### `make_lmdb_mixed_batch_collate(graph_config)`

这个函数包一层 collate:

```text
_collate_lmdb_mixed_batch(batch)
  -> build_precomputed_flat_graph(...)
  -> result.update(graph_data)
```

传给 `build_precomputed_flat_graph()` 的输入:

```python
result["coord"]  # [total_atoms, 3]
result["atype"]  # [total_atoms]
result["batch"]  # [total_atoms]
result["ptr"]    # [nframes + 1]
rcut, sel, a_rcut, a_sel, mixed_types
box=result.get("box")
```

## flat graph 预计算

实现位置: `deepmd/pt/utils/nlist.py::build_precomputed_flat_graph()`

它按 frame 循环处理, 因为每个 frame 的 `nloc` 和 ghost 扩展数量都可能不同。

### 每个 frame 内部

对 frame `i`:

```text
start = ptr[i]
end = ptr[i+1]
nloc = end - start
frame_coord = coord[start:end].reshape(1, nloc, 3)
frame_atype = atype[start:end].reshape(1, nloc)
frame_box = box[i:i+1]
```

然后:

1. 如果有 box, 先 normalize 到周期盒内。
2. 调 `extend_coord_with_ghosts_with_images()` 生成:
   - `frame_extended_coord`
   - `frame_extended_atype`
   - `frame_mapping`
   - `frame_extended_image`
3. 调 `build_neighbor_list()` 生成 frame 内 extended index 空间下的 `frame_nlist_ext`。
4. 把 frame 内 index 加上 `extended_offset`, 转成整个 mixed batch 的全局 extended index。

### 拼成 batch 级 flat graph

所有 frame 循环结束后:

- `extended_coord`: `[total_extended_atoms, 3]`
- `extended_atype`: `[total_extended_atoms]`
- `extended_batch`: `[total_extended_atoms]`, extended atom -> frame
- `extended_image`: `[total_extended_atoms, 3]`
- `mapping`: `[total_extended_atoms]`, extended atom -> flattened local atom index
- `central_ext_index`: `[total_atoms]`, local atom 对应的 extended atom index
- `nlist_ext`: `[total_atoms, nnei]`, 邻居指向 extended atom index, padding 为 `-1`
- `nlist`: `[total_atoms, nnei]`, 邻居映射回 flattened local atom index
- `nlist_mask`: `[total_atoms, nnei]`

其中 `nlist` 是通过:

```text
nlist = mapping[nlist_ext_clamped]
```

把 extended neighbor index 转回 local flat atom index。

### angle neighbor 和 dynamic graph index

`build_precomputed_flat_graph()` 用 edge neighbor 距离筛出 angle neighbor:

```text
a_dist_mask = (dist[:, :a_sel] < a_rcut) & nlist_mask[:, :a_sel]
a_nlist_ext = where(a_dist_mask, nlist_ext[:, :a_sel], -1)
a_nlist = mapping[a_nlist_ext_clamped]
a_nlist_mask = a_nlist_ext >= 0
```

然后:

```python
edge_index, angle_index = get_graph_index_flat(nlist, a_nlist_mask)
```

这两个 index 供 dynamic selection 的 RepFlowLayer 使用。

collate 最终会额外返回这些 graph fields:

```text
extended_atype
extended_batch
extended_image
extended_ptr
mapping
central_ext_index
nlist
nlist_ext
a_nlist
a_nlist_ext
nlist_mask
a_nlist_mask
edge_index
angle_index
```

## batch 从 DataLoader 到模型输入

`Trainer.get_data()` 检测:

```python
is_mixed_batch = "batch" in batch_data and "ptr" in batch_data
```

普通 tensor 全部 `.to(DEVICE)`。然后:

```python
if is_mixed_batch:
    input_keys += _FLAT_GRAPH_INPUT_KEYS
    batch_data["batch"] = batch_data["batch"].to(DEVICE)
    batch_data["ptr"] = batch_data["ptr"].to(DEVICE)
```

`_FLAT_GRAPH_INPUT_KEYS` 包括:

```text
batch, ptr,
extended_atype, extended_batch, extended_image, extended_ptr,
mapping, central_ext_index,
nlist, nlist_ext, a_nlist, a_nlist_ext,
nlist_mask, a_nlist_mask,
edge_index, angle_index
```

所以 mixed batch 的 `input_dict` 不是普通:

```text
coord, atype, box, fparam, aparam
```

而是:

```text
coord, atype, box, fparam, aparam
+ flat graph fields
```

label 里仍保留 `energy`, `force`, `virial` 等。由于 force 是 atom-wise key, label force 是 `[total_atoms, 3]`。

## `ModelWrapper.forward()` 如何转发

`deepmd/pt/train/wrapper.py::ModelWrapper.forward()` 接收新增参数:

```text
batch, ptr,
extended_atype, extended_batch, extended_image, extended_ptr,
mapping, central_ext_index,
nlist, nlist_ext, a_nlist, a_nlist_ext,
nlist_mask, a_nlist_mask,
edge_index, angle_index
```

只要 `batch is not None and ptr is not None`, 就把这些 flat graph fields 放进 `input_dict`。训练时:

```text
EnergyStdLoss.forward(input_dict, model, label, natoms=...)
  -> model(**input_dict)
```

这里 `natoms` 在 mixed batch 下是 `atype.shape[0]`, 也就是 flattened total atoms。

## 模型 forward 如何选择 flat 路径

模型类由 `deepmd/pt/model/model/make_model.py` 动态生成。`forward()` 没有单独判断 mixed batch, 但 `forward_common()` 接收到了 `batch` 和 `ptr` 后会进入 flat 方法:

```text
model.forward(...)
  -> forward_common(..., batch, ptr, graph_fields)
  -> forward_common_flat(...)
  -> forward_common_flat_native(...)
```

`forward_common_flat_native()` 做几件关键事:

1. `_input_type_cast(coord, box, fparam, aparam)`。
2. 如果要算 force, `coord = coord.clone().detach().requires_grad_(True)`。
3. 如果要算 virial 且有 box, `box.requires_grad_(True)`。
4. 用 `rebuild_extended_coord_from_flat_graph()` 根据 flattened `coord`, `box`, `mapping`, `extended_batch`, `extended_image` 重建 `extended_coord`。
5. 调 `forward_common_lower_flat(...)`。
6. 如果需要力或 virial, 调 `_compute_derivatives_flat(...)`。

为什么要重建 `extended_coord`: collate 阶段预计算的是 graph topology 和 ghost image 信息, 训练时坐标需要参与 autograd。用当前 requires-grad 的 `coord` 重新根据 `mapping` 和 `extended_image` 生成 `extended_coord`, 可以保留梯度链。

## lower flat forward

`forward_common_lower_flat()`:

```text
self.atomic_model.forward_common_atomic_flat(
    extended_coord,
    extended_atype,
    extended_batch,
    nlist,
    mapping,
    batch,
    ptr,
    fparam,
    aparam,
    graph_fields...
)
```

atomic model 返回 atom-wise 输出后, 如果有 `"energy"`:

```python
energy_redu = zeros([nframes, energy_dim])
energy_redu.index_add_(0, batch, energy_atomic)
model_ret["energy_redu"] = energy_redu
```

也就是说 mixed batch 的 frame energy reduce 不依赖 dense `[nframes, max_nloc]`, 而是用 `batch` 对 flat atom energy 做 scatter-add。

## flat atomic model

位置: `deepmd/pt/model/atomic_model/dp_atomic_model.py::forward_common_atomic_flat()`

主要流程:

```text
extended_coord [total_extended_atoms, 3]
extended_atype [total_extended_atoms]
extended_batch [total_extended_atoms]
nlist [total_atoms, nnei]
mapping [total_extended_atoms]
batch [total_atoms]
ptr [nframes + 1]
  -> descriptor.forward_flat(...)
  -> fitting_net.forward_flat(...)
  -> apply_out_stat(...)
  -> atom mask
  -> return flat fit_ret
```

### descriptor.forward_flat

DPA3 的 `forward_flat()`:

1. `extended_coord` cast 到 descriptor 精度。
2. 对所有 extended atoms 做 type embedding:

```python
node_ebd_ext = self.type_embedding(extended_atype)
```

3. `central_ext_index` 如果没有则从 `extended_batch` 和 `ptr` 计算。
4. 取中心原子 embedding:

```python
node_ebd_inp = node_ebd_ext[central_ext_index]
```

5. 调:

```text
self.repflows.forward_flat(
    nlist,
    extended_coord,
    extended_atype,
    extended_batch,
    node_ebd_ext,
    mapping,
    batch,
    ptr,
    graph_fields...
)
```

返回:

- `descriptor`: `[total_atoms, descriptor_dim]`
- `rot_mat`: `[total_atoms, e_dim, 3]`
- `g2`, `h2`

### RepFlows.forward_flat

`Repflows.forward_flat()` 是 flat graph 的 descriptor 主体:

1. 要求 collate 已经给出 `central_ext_index`, `nlist_ext`, `a_nlist`, `a_nlist_ext`, `nlist_mask`, `a_nlist_mask`。
2. 用 `prod_env_mat_flat()` 在 extended atom index 空间里计算 edge environment:

```text
extended_coord [total_extended_atoms, 3]
nlist_ext [total_atoms, nnei]
coord_central = extended_coord[central_ext_index]
  -> dmatrix, diff, sw
```

3. 用 `a_nlist_ext` 计算 angle environment。
4. 用 `extended_atype_embd[central_ext_index]` 得到中心原子 type embedding。
5. 如果 `use_dynamic_sel=True`, 用预计算的 `edge_index`, `angle_index`, `nlist_mask`, `a_nlist_mask` 把 edge/angle 张量 flatten 到 dynamic graph 格式。
6. 给每层 `RepFlowLayer` 调普通 `ll.forward(...)`, 但包装成一个 synthetic one-frame batch:

```text
node_ebd_batched = node_ebd.unsqueeze(0)      # [1, total_atoms, n_dim]
nlist_batched = nlist.unsqueeze(0)            # [1, total_atoms, nnei]
a_nlist_batched = a_nlist.unsqueeze(0)        # [1, total_atoms, a_sel]
```

dynamic selection 下 edge/angle 本来就是 flat 的, 不额外加 batch 维。

7. 最后用 `_cal_hg_dynamic()` 或 `_cal_hg()` 计算 `rot_mat`, 再 squeeze 回 flat:

```text
node_ebd [total_atoms, n_dim]
rot_mat [total_atoms, e_dim, 3]
```

这里的关键点是: RepFlowLayer 本身不需要知道真实 batch 中每帧有多少 atom。flat path 把所有原子看成一个 synthetic frame, 但 graph/nlist 已经保证不会跨真实 frame 连边。

## fitting_net.forward_flat

位置: `deepmd/pt/model/task/invar_fitting.py::forward_flat()`

虽然 descriptor 是 flat 的, fitting 仍复用原有 dense fitting:

1. 根据 `ptr` 计算:

```text
nframes = len(ptr) - 1
atom_counts = ptr[1:] - ptr[:-1]
max_nloc = max(atom_counts)
local_index = arange(total_atoms) - ptr[batch]
```

2. 分配 dense buffer:

```text
descriptor_batch [nframes, max_nloc, descriptor_dim]
atype_batch      [nframes, max_nloc], padding type = -1
gr_batch         [nframes, max_nloc, ...] if needed
aparam_batch     [nframes, max_nloc, ...] if needed
```

3. scatter flat 输入:

```python
descriptor_batch[batch, local_index] = descriptor
atype_batch[batch, local_index] = atype
```

4. 调原有 `self.forward(...)`。
5. 用:

```python
valid_atom_mask = arange(max_nloc)[None, :] < atom_counts[:, None]
```

把 atom-wise 输出 gather 回 flat:

```python
result_flat[key] = value[valid_atom_mask]
```

所以当前设计是:

```text
descriptor/neighbor/force graph: flat
fitting MLP: 临时 dense, 复用旧实现
输出: flat
```

## force 和 virial

`_compute_derivatives_flat()`:

### Force

```python
energy_atomic = fit_ret["energy"]  # [total_atoms, 1]
energy_derv_r = autograd.grad(
    outputs=energy_atomic.sum(),
    inputs=coord,
    create_graph=True,
    retain_graph=True,
)[0]  # [total_atoms, 3]
fit_ret["energy_derv_r"] = -energy_derv_r.unsqueeze(-2)
fit_ret["dforce"] = -energy_derv_r
```

这会直接得到 flattened force, shape `[total_atoms, 3]`, 和 label force 的 flat shape 对齐。

### Virial

如果需要 cell gradient 且 box 不为 None:

```python
energy_redu = fit_ret["energy_redu"]  # [nframes, 1]
energy_derv_c_redu = autograd.grad(
    outputs=energy_redu.sum(),
    inputs=box,
    create_graph=True,
    retain_graph=True,
)[0]  # [nframes, 9]
fit_ret["energy_derv_c_redu"] = energy_derv_c_redu.unsqueeze(1)
```

atomic virial 在 flat mixed batch 下显式不支持。

## loss 如何适配 mixed batch

`EnergyStdLoss.forward()` 检测:

```python
is_mixed_batch = "ptr" in input_dict and input_dict["ptr"] is not None
```

mixed batch 下:

```text
natoms_per_frame = ptr[1:] - ptr[:-1]
atom_norms = 1.0 / natoms_per_frame
```

energy / virial 的归一化使用 frame-wise atom count:

```python
get_frame_norm(value) -> atom_norms.view([-1] + [1] * (value.dim() - 1))
weighted_mean(value, power) -> mean(value * frame_norm**power)
normalized_rmse(diff) -> sqrt(mean((diff * frame_norm)**2))
```

force loss 不需要 frame-wise reshape, 直接对 flattened force 做 `reshape(-1)` 求 MSE/MAE。

限制:

- generalized force loss 仍使用 `natoms` reshape 成 `[-1, natoms*3]`, 对 mixed batch 不天然适配。
- atomic virial 未实现。
- 当前主要适合 energy/force/virial 标准能量模型训练。

## 完整调用图

```text
dp --pt train test_mptraj/lmdb_mixed_batch.json
  -> Trainer.__init__
       -> get_data_loader(LmdbDataset.mixed_batch=True)
            -> DataLoader(batch_size=_data.batch_size,
                          collate_fn=make_lmdb_mixed_batch_collate(graph_config))
  -> Trainer.run.step
       -> get_data()
            -> batch_data = next(training_data)
            -> detect batch + ptr
            -> move tensors to DEVICE
            -> input_dict includes coord/atype/box + flat graph fields
       -> ModelWrapper.forward(...)
            -> input_dict.update(flat graph fields)
            -> EnergyStdLoss.forward(input_dict, model, label, natoms=total_atoms)
                 -> model(**input_dict)
                      -> forward_common_flat(...)
                           -> rebuild_extended_coord_from_flat_graph(...)
                           -> forward_common_lower_flat(...)
                                -> atomic_model.forward_common_atomic_flat(...)
                                     -> descriptor.forward_flat(...)
                                          -> DPA3.forward_flat(...)
                                               -> Repflows.forward_flat(...)
                                                    -> prod_env_mat_flat(...)
                                                    -> RepFlowLayer.forward(...)
                                     -> fitting_net.forward_flat(...)
                                          -> scatter flat -> dense [nframes, max_nloc]
                                          -> old fitting forward
                                          -> gather dense -> flat
                                     -> apply_out_stat + atom mask
                                -> energy_redu.index_add_(0, batch, atom_energy)
                           -> autograd force/virial if needed
                 -> EnergyStdLoss mixed-batch normalization with ptr
       -> backward + optimizer step
```

## 和普通 LMDB same-nloc batch 的差异

普通 `mixed_batch=False`:

- sampler 保证一个 batch 内所有 frame 的 `nloc` 相同。
- collate 可以 stack 成 `[nframes, nloc, ...]`。
- 模型走原有 dense forward。

mixed `mixed_batch=True`:

- sampler 可以随机抽不同 `nloc` 的 frame。
- atom-wise label/input flatten 成 `[total_atoms, ...]`。
- `batch` 和 `ptr` 记录 frame 边界。
- graph 在 collate 里预计算, 避免模型内部再按 dense batch 建图。
- descriptor 走 flat forward。
- fitting net 临时 dense 化后再 flatten。
- energy 用 `index_add_` 按 frame reduce。

## 当前实现的关键假设和限制

1. mixed batch 当前要求 DPA3/RepFlow 这类带 `repflows` 的 flat-graph capable descriptor。
2. collate 预计算的 graph topology 基于当前 batch 坐标和 box; forward 会重建 `extended_coord` 以保留 autograd。
3. flat path 通过 synthetic one-frame batch 调 RepFlowLayer, 依赖 nlist/edge_index 不跨真实 frame。
4. `filter:N` 与 `mixed_batch=True` 不兼容, 因为 mixed fast path 跳过了 nloc 扫描。
5. `auto_prob_style/block weighting` 与 `mixed_batch=True` 也还不支持。
6. atomic virial 未实现。
7. generalized force loss 对 mixed nloc 仍有 reshape 假设, 需要单独改造。
