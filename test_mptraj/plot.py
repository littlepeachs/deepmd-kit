import numpy as np
import matplotlib.pyplot as plt
date = "0619"
task_name = "mptraj"
data = np.genfromtxt(f"./lcurve.out", names=True)
data2 = np.genfromtxt(f"./lcurve_esen.out", names=True)

# task_name = "MPTraj"
# data = np.genfromtxt(f"./0420-DPA3RBF_Dynamics_MPTraj_Base-l6-nd128-ed64-ad32-rbfF-slr0.001-llr1e-05-spe0.2-lpe20-spf100-lpf20-spv0.02-lpv1-bsauto:256-steps1e+06/MPTraj/lcurve.out", names=True)
# data2 = np.genfromtxt(f"./0422-DPA3_MPTraj_RBF_Torsion_Node-l6-nd128-ed64-ad32-rbfT-slr0.001-llr1e-05-spe0.2-lpe20-spf100-lpf20-spv0.02-lpv1-bsauto:256-steps1e+06/MPTraj/lcurve.out", names=True)
# task_name = "Spice"
# data = np.genfromtxt(f"./0505-DPA3_Spice_Base-l6-nd128-ed64-ad32-rbfF-torF-slr0.001-llr1e-05-spe0.2-lpe20-spf100-lpf20-spv0.02-lpv1-bsauto:256-steps1e+06/{task_name}/lcurve.out", names=True)

# data2 = np.genfromtxt(f"./0505-DPA3_Spice_RBF_Torsion-l6-nd128-ed64-ad32-rbfT-torT-slr0.001-llr1e-05-spe0.2-lpe20-spf100-lpf20-spv0.02-lpv1-bsauto:256-steps1e+06/{task_name}/lcurve.out", names=True)
# 计算lcurve_rbf_eachlayer在其他两个底下的比例（仅前1000000步）
common_steps = []
below_count_1 = 0
below_count_2 = 0
total_count = 0

# 找到三个数据集中共同的步骤（限制在前1000000步）
for step2 in data2["step"]:
    if step2 <= 4000000 and step2 in data["step"]:
        common_steps.append(step2)

# 对于每个共同的步骤，检查lcurve_rbf_eachlayer的值是否低于其他两个
for name in data.dtype.names[1:2]:
    for step in common_steps:
        idx = np.where(data["step"] == step)[0][0]
        # idx1 = np.where(data1["step"] == step)[0][0]
        idx2 = np.where(data2["step"] == step)[0][0]
        # 如果data2[name]的值超过500，将其设置为10
        # if data2[name][idx2] > 500:
        #     data2[name][idx2] = 10
        if data2[name][idx2] < data[name][idx]:
            below_count_1 += 1
        # if data2[name][idx2] < data1[name][idx1]:
            # below_count_2 += 1
        total_count += 1

below_ratio_norbf = below_count_1 / total_count if total_count > 0 else 0

print(f"lcurve_rbf_eachlayer在lcurve1底下点的比例(前1000000步): {below_ratio_norbf:.2%}")

# 绘制图表（仅前1000000步）
for name in data.dtype.names[1:2]:
    # 筛选前1000000步的数据
    mask = data["step"] <= 4000000
    # mask1 = data1["step"] <= 1000000
    mask2 = data2["step"] <= 4000000
    
    plt.plot(data["step"][mask], data[name][mask], label=name+" (baseline)",alpha=0.6)
    # plt.plot(data1["step"][mask1], data1[name][mask1], label=name+" (no angle)")
    plt.plot(data2["step"][mask2], data2[name][mask2], label=name+" (eqnorm)",alpha=0.6)
plt.legend()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.xscale("symlog")
plt.yscale("log")
plt.grid()
plt.title(f"1000000 - baseline with eqnorm mptraj: {below_ratio_norbf:.2%}")
plt.savefig(f"lcurve.png")