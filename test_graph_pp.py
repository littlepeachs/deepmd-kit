# test_graph_partition_main.py
# 用法：
#   1) 确保你的实现文件可 import，比如 graph_parallel.py
#   2) 运行：python test_graph_partition_main.py

import copy
import sys
import traceback

import torch

import graph_parallel as m  # <-- 改成你的模块名


class FakeDistUtils:
    """可控的 distutils mock，用于模拟 graph-parallel 环境。"""

    def __init__(self, world_size=1, rank=0, initialized=True):
        self._world_size = world_size
        self._rank = rank
        self._initialized = initialized

        self.gather_called = False
        self.reduce_called = False

    def initialized(self):
        return self._initialized

    def get_gp_world_size(self):
        return self._world_size

    def get_gp_rank(self):
        return self._rank

    def gather_from_model_parallel_region_sum_grad(self, tensor, dim=0):
        self.gather_called = True
        return tensor

    def reduce_from_model_parallel_region(self, tensor):
        self.reduce_called = True
        return tensor


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def assert_equal(a, b, msg: str = ""):
    if a != b:
        raise AssertionError(msg or f"Expected {b}, got {a}")


def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, msg: str = ""):
    if not torch.equal(a, b):
        raise AssertionError(msg or f"Tensor not equal.\nA={a}\nB={b}")


def make_batch_graph(num_nodes=10, num_heads=2, device="cpu"):
    atomic_numbers = torch.arange(num_nodes, device=device, dtype=torch.long)
    node_batch = torch.zeros(num_nodes, device=device, dtype=torch.long)

    # attn_mask / angle_embedding: flatten 维度 = num_nodes * num_heads
    attn_mask = torch.arange(num_nodes * num_heads, device=device, dtype=torch.long)
    angle_embedding = torch.arange(num_nodes * num_heads, device=device, dtype=torch.float32)

    neighbor_list = torch.arange(num_nodes, device=device, dtype=torch.long).view(num_nodes, 1)
    neighbor_mask = torch.ones(num_nodes, device=device, dtype=torch.bool)

    return {
        "atomic_numbers": atomic_numbers,
        "node_batch": node_batch,
        "attn_mask": attn_mask,
        "angle_embedding": angle_embedding,
        "neighbor_list": neighbor_list,
        "neighbor_mask": neighbor_mask,
        "padding_atoms": 123,  # 会被改成 local_size
        "smooth_two_body": torch.randn(num_nodes, 4, device=device),
        # 可选：如果你代码里还会保存 batch_cart_coords_full，可以加上
        "batch_cart_coords": torch.randn(num_nodes, 3, device=device),
    }


def expected_partition(num_nodes: int, world_size: int):
    # 复刻 _balanced_partition_sizes 的逻辑，用于测试期望
    base = num_nodes // world_size
    rem = num_nodes % world_size
    sizes = [base + (1 if i < rem else 0) for i in range(world_size)]
    offsets = []
    s = 0
    for i in range(world_size):
        offsets.append(s)
        s += sizes[i]
    return sizes, offsets


def run_partition_batch_graph_core_tests():
    print("[TEST] partition_batch_graph core slicing")

    # 1) 注入 fake distutils（模拟 3 卡 graph parallel）
    fake = FakeDistUtils(world_size=3, rank=0, initialized=True)
    m.distutils = fake  # 直接替换模块变量

    # 2) 准备输入
    num_nodes = 10
    num_heads = 2
    bg = make_batch_graph(num_nodes=num_nodes, num_heads=num_heads)
    bg_full = copy.deepcopy(bg)  # 用于校验 full 备份字段
    sizes, offsets = expected_partition(num_nodes, fake.get_gp_world_size())
    assert_equal(sizes, [4, 3, 3], "Partition sizes mismatch for num_nodes=10, world_size=3")

    # 3) 逐 rank 测试切分结果
    for rank in range(fake.get_gp_world_size()):
        fake._rank = rank

        local_bg = copy.deepcopy(bg)  # 每个 rank 单独跑一次
        out = m.partition_batch_graph(local_bg)

        local_size = sizes[rank]
        node_offset = offsets[rank]
        expected_nodes = list(range(node_offset, node_offset + local_size))

        # 3.1 atomic_numbers 连续切片
        assert_equal(out["atomic_numbers"].tolist(), expected_nodes,
                     f"rank={rank}: atomic_numbers slice mismatch")

        # 3.2 padding_atoms 改成 local_size
        assert_equal(out["padding_atoms"], local_size,
                     f"rank={rank}: padding_atoms should be local_size")

        # 3.3 full 备份字段存在并与原始全量一致
        assert_true("node_batch_full" in out, f"rank={rank}: missing node_batch_full")
        assert_true("atomic_numbers_full" in out, f"rank={rank}: missing atomic_numbers_full")
        assert_true("batch_cart_coords_full" in out, f"rank={rank}: missing batch_cart_coords_full")

        assert_tensor_equal(out["node_batch_full"], bg_full["node_batch"],
                            f"rank={rank}: node_batch_full mismatch")
        assert_tensor_equal(out["atomic_numbers_full"], bg_full["atomic_numbers"],
                            f"rank={rank}: atomic_numbers_full mismatch")
        assert_tensor_equal(out["batch_cart_coords_full"], bg_full["batch_cart_coords"],
                            f"rank={rank}: batch_cart_coords_full mismatch")

        # 3.4 node-level tensor 的第0维匹配 local_size
        assert_equal(out["smooth_two_body"].shape[0], local_size,
                     f"rank={rank}: smooth_two_body first dim mismatch")
        assert_equal(out["neighbor_list"].shape[0], local_size,
                     f"rank={rank}: neighbor_list first dim mismatch")
        assert_equal(out["neighbor_mask"].shape[0], local_size,
                     f"rank={rank}: neighbor_mask first dim mismatch")
        assert_equal(out["node_batch"].shape[0], local_size,
                     f"rank={rank}: node_batch first dim mismatch")

        # 3.5 attention flatten 切分（第0维 = local_nodes * num_heads）
        assert_equal(out["attn_mask"].shape[0], local_size * num_heads,
                     f"rank={rank}: attn_mask length mismatch")
        assert_equal(out["angle_embedding"].shape[0], local_size * num_heads,
                     f"rank={rank}: angle_embedding length mismatch")

        start = node_offset * num_heads
        end = start + local_size * num_heads
        assert_equal(out["attn_mask"].tolist(), list(range(start, end)),
                     f"rank={rank}: attn_mask slice values mismatch")

    print("[PASS] partition_batch_graph core slicing OK")


def run_partition_batch_graph_edge_tests():
    print("[TEST] partition_batch_graph edge cases")

    # 注入 fake distutils
    fake = FakeDistUtils(world_size=3, rank=0, initialized=True)
    m.distutils = fake

    # case 1: 未初始化 => 原样返回
    fake._initialized = False
    bg = make_batch_graph(10, 2)
    out = m.partition_batch_graph(bg)
    assert_true(out is bg, "When dist not initialized, should return same dict")
    assert_tensor_equal(out["atomic_numbers"], torch.arange(10, dtype=torch.long),
                        "When dist not initialized, atomic_numbers should be unchanged")

    # case 2: world_size==1 => 原样返回
    fake._initialized = True
    fake._world_size = 1
    bg = make_batch_graph(10, 2)
    out = m.partition_batch_graph(bg)
    assert_true(out is bg, "When world_size==1, should return same dict")

    # case 3: num_nodes==0 => 原样返回（你的实现是直接 return）
    fake._world_size = 3
    bg = make_batch_graph(0, 2)
    out = m.partition_batch_graph(bg)
    assert_true(out is bg, "When num_nodes==0, should return same dict")
    assert_equal(out["atomic_numbers"].numel(), 0, "num_nodes==0 should keep empty")

    # case 4: 缺少 atomic_numbers => KeyError
    threw = False
    try:
        m.partition_batch_graph({"node_batch": torch.zeros(3, dtype=torch.long)})
    except KeyError:
        threw = True
    assert_true(threw, "Missing atomic_numbers should raise KeyError")

    print("[PASS] edge cases OK")


def main():
    try:
        run_partition_batch_graph_core_tests()
        run_partition_batch_graph_edge_tests()
    except Exception as e:
        print("\n[FAIL]", str(e))
        traceback.print_exc()
        sys.exit(1)

    print("\nALL TESTS PASSED ✅")


if __name__ == "__main__":
    main()
