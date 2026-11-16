# summarize_and_plot.py  (替换你的 summarize_and_plot)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator

def summarize_and_plot(
    csv_path,
    out_dir="results/figs",
    ns=(10000, 100000),
    title_str="Power Posterior 对比：不同 beta、不同 n",
    use_log_mmd=False,                    # 如果 mmd 有离群尖峰，可以 True（log 轴下限会自动>0）
    dedup_key=("method", "n", "beta", "seed"),  # “同名”键；如要忽略 seed：("method","n","beta")
):
    assert len(ns) == 2, "ns 需要两个取值，例如 (10000, 100000)"
    n_left, n_right = ns

    # ---- 美观的全局样式 ----
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titleweight": "bold",
    })

    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读取 & 去重（保留最新）----
    df = pd.read_csv(csv_path)
    before = len(df)
    df = _dedup_keep_latest(df, key_cols=dedup_key)
    after = len(df)
    print(f"[dedup] {before} -> {after} rows (keep latest per {dedup_key})")

    # 保存去重后的 CSV 以便复现
    clean_csv = out_dir / "metrics_clean_latest.csv"
    df.to_csv(clean_csv, index=False)
    print(f"Saved cleaned CSV: {clean_csv}")

    # ---- 类型与必要列 ----
    for col in ["n", "beta", "c2st", "mmd"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 只保留绘图需要的完整行
    need_cols = ["method", "n", "beta", "c2st", "mmd"]
    df = df.dropna(subset=[c for c in need_cols if c in df.columns])

    # ---- 聚合统计：mean/std（跨 seed 或重复运行）----
    agg = (
        df.groupby(["method", "n", "beta"], as_index=False)
          .agg(c2st_mean=("c2st", "mean"), c2st_std=("c2st", "std"),
               mmd_mean=("mmd", "mean"),   mmd_std=("mmd", "std"),
               count=("c2st", "size"))
          .sort_values(["method", "n", "beta"])
    )
    summary_csv = out_dir / "summary.csv"
    agg.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    # ---- 配色与标记：同 method 保持一致 ----
    methods = list(agg["method"].dropna().unique())
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<"]
    color_map  = {m: colors[i % len(colors)]   for i, m in enumerate(methods)}
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(methods)}

    # ---- 公共美化 ----
    def _beautify(ax):
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))

    # ---- 作图：带“夹到边界”的固定 y 轴 ----
    def plot_metric(ax, n_value, metric_key, title, y_range=None, logy=False):
        subn = agg[agg["n"] == n_value]
        # 范围处理
        if y_range is not None:
            lo, hi = y_range
            if logy and lo <= 0:
                lo = 1e-8  # log 轴的下界必须>0
        else:
            lo = hi = None

        for method, sub in subn.groupby("method"):
            sub = sub.sort_values("beta")

            y_orig = sub[metric_key].to_numpy()
            std_key = metric_key.replace("_mean", "_std")
            ystd = sub[std_key].fillna(0).to_numpy() if std_key in sub else np.zeros_like(y_orig)

            if y_range is not None:
                y_plot   = np.clip(y_orig,        lo, hi)
                lower_bd = np.clip(y_orig - ystd, lo, hi)
                upper_bd = np.clip(y_orig + ystd, lo, hi)
            else:
                y_plot, lower_bd, upper_bd = y_orig, y_orig - ystd, y_orig + ystd

            ax.plot(
                sub["beta"], y_plot,
                label=method,
                color=color_map[method],
                marker=marker_map[method],
                markersize=5.5,
                linewidth=2.2,
            )
            if np.isfinite(ystd).any():
                ax.fill_between(
                    sub["beta"], lower_bd, upper_bd,
                    alpha=0.18, color=color_map[method], linewidth=0
                )

        ax.set_xlabel("beta")
        ax.set_title(title)

        if y_range is not None:
            ax.set_ylim((lo, hi) if not logy else (max(lo, 1e-8), hi))

        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(ScalarFormatter())

        _beautify(ax)

    # ---- 画 1x4 面板 ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharex=False)
    plt.subplots_adjust(bottom=0.25, wspace=0.28, top=0.82)

    c2st_range = (0.5, 1.0)
    mmd_range  = (1e-8, 0.05) if use_log_mmd else (0.0, 0.05)

    # 顺序：n_left 的 c2st / mmd，再到 n_right 的 c2st / mmd
    plot_metric(axes[0], n_left,  "c2st_mean", f"c2st  (n={n_left})",  y_range=c2st_range, logy=False)
    plot_metric(axes[1], n_left,  "mmd_mean",  f"mmd   (n={n_left})",  y_range=mmd_range,  logy=use_log_mmd)
    plot_metric(axes[2], n_right, "c2st_mean", f"c2st  (n={n_right})", y_range=c2st_range, logy=False)
    plot_metric(axes[3], n_right, "mmd_mean",  f"mmd   (n={n_right})", y_range=mmd_range,  logy=use_log_mmd)

    # 统一底部图例
    proxy = [Line2D([0],[0], color=color_map[m], marker=marker_map[m], lw=2.2, label=m) for m in methods]
    if proxy:
        fig.legend(
            handles=proxy,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(proxy), 6),
            frameon=False
        )

    fig.suptitle(title_str, fontsize=15, fontweight="bold")
    out_png = out_dir / f"panel_pretty_1x4_{n_left}_{n_right}.png"
    out_svg = out_dir / f"panel_pretty_1x4_{n_left}_{n_right}.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")  # 矢量图，论文更清晰
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

def _dedup_keep_latest(
    df: pd.DataFrame,
    key_cols=("method", "n", "beta", "seed"),   # 需要的话改成 ("method","n","beta")
    time_col="timestamp",
):
    df = df.copy()

    # 统一数值类型，避免 "10_000" 字符串之类造成分组错位
    for c in ("n", "beta", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 轻微清洗 method：把孤立写成 "NlE"/"nLe" 的改成 "NLE"；其他如 "RouteB_NLE" 保持不变
    if "method" in df.columns:
        df["method"] = df["method"].apply(
            lambda s: ("NLE" if isinstance(s, str) and s.strip().lower() == "nle"
                       else ("NRE" if isinstance(s, str) and s.strip().lower() == "nre"
                             else s))
        )

    # 解析时间；无时区也没关系，比较相对顺序即可
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
    else:
        # 没时间列就用行序兜底
        t = pd.Series(pd.NaT, index=df.index)

    df["__t__"] = t
    df["__row__"] = np.arange(len(df))  # 保证稳定排序

    # 先按时间、再按行序排序，确保“最新在最后”
    df = df.sort_values(["__t__", "__row__"])

    # 对键去重并保留最后一条（最新）
    out = df.drop_duplicates(subset=list(key_cols), keep="last").drop(columns=["__t__", "__row__"])

    return out




if __name__ == "__main__":
    name = ["gaussian_mixture", "two_moons", "slcp", "sir"]
    number = 1
    save_dir = Path(name[number])
    csv_path = save_dir / "metrics.csv"

    df = pd.read_csv(csv_path)

    df_before = len(df)
    df = _dedup_keep_latest(df, key_cols=("method", "n", "beta", "seed"), time_col="timestamp")
    print(f"[dedup] {df_before} -> {len(df)} rows after keep-latest per (method,n,beta,seed)")

    for col in ["n", "beta", "c2st", "mmd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summarize_and_plot(
        csv_path= csv_path,
        out_dir= save_dir/"figs",
        ns=(10000, 100000),
        title_str="Power Posterior ",
        use_log_mmd=False,
        dedup_key=("method", "n", "beta", "seed")
    )
