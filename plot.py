import pandas as pd
import matplotlib.pyplot as plt


def load_lcurve(path):
    cols = [
        "step",
        "rmse_val", "rmse_trn",
        "rmse_e_val", "rmse_e_trn",
        "rmse_f_val", "rmse_f_trn",
        "rmse_v_val", "rmse_v_trn",
        "lr",
    ]
    df = pd.read_csv(
        path,
        comment="#",
        delim_whitespace=True,
        header=None,
        names=cols,
    )
    df = df.dropna()
    return df


def plot_rmse_val_compare(df1, df2, label1="gp", label2="single"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].plot(df2["step"], df2["rmse_val"], label=label2, lw=1.8, alpha=0.6)
    axes[0].set_title(f"{label2} rmse_val")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("rmse_val")
    axes[0].set_yscale("log", base=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df1["step"], df1["rmse_val"], label=label1, lw=1.8, alpha=0.6)
    axes[1].set_title(f"{label1} rmse_val")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log", base=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("RMSE val comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("rmse_val_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    df_gp = load_lcurve("lcurve_4rank.out")
    df_single = load_lcurve("lcurve_single.out")
    plot_rmse_val_compare(df_gp, df_single, label1="dp_gp", label2="single")