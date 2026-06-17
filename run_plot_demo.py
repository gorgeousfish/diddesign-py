import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, "/Users/cxy/Desktop/2026project/pydiddesign/diddesign-py/src")

from diddesign import did, did_check
from diddesign.visualization import plot_estimates, plot_trends, plot_placebo, plot_diagnostics, plot_pattern

output_dir = "/Users/cxy/Desktop/2026project/pydiddesign/diddesign-py"

# ============================================================
# 示例1: 标准DID - 带显著处理效应的面板数据
# ============================================================
np.random.seed(2024)
n = 150
data1 = pd.DataFrame({
    "unit": np.repeat(range(n), 5),
    "time": np.tile([2018, 2019, 2020, 2021, 2022], n),
    "treat": np.where(
        (np.repeat(range(n), 5) < n//2) & (np.tile([2018,2019,2020,2021,2022], n) >= 2021), 1, 0
    ),
    "y": (
        np.repeat(np.random.normal(10, 2, n), 5)  # unit FE
        + np.tile([0, 0.5, 1.0, 1.5, 2.0], n)  # time trend
        + np.where(
            (np.repeat(range(n), 5) < n//2) & (np.tile([2018,2019,2020,2021,2022], n) >= 2021),
            3.0, 0  # ATT = 3.0
        )
        + np.random.normal(0, 0.8, n*5)  # noise
    ),
})

print("Running did()...")
result1 = did(data1, formula="y ~ treat", time="time", unit_id="unit", n_boot=100, random_seed=42)
print("Running did_check()...")
check1 = did_check(data1, formula="y ~ treat", time="time", unit_id="unit", n_boot=100, random_seed=42)

# 图1: 效应估计图（含placebo）
print("Generating plot 1: estimates with placebo...")
fig1 = plot_estimates(result1, check_fit=check1, 
                      title="Treatment Effect Estimates",
                      ylabel="Estimated ATT",
                      style="publication", show=False, dpi=200,
                      save=f"{output_dir}/example_plot_estimates.png")
print(f"Saved: example_plot_estimates.png")

# 图2: 趋势图
print("Generating plot 2: trends...")
fig2 = plot_trends(check1, ci=True,
                   title="Outcome Trends: Treated vs Control",
                   style="publication", show=False, dpi=200,
                   save=f"{output_dir}/example_plot_trends.png")
print(f"Saved: example_plot_trends.png")

# 图3: Placebo检验图
print("Generating plot 3: placebo...")
fig3 = plot_placebo(check1,
                    title="Pre-treatment Placebo Test",
                    style="publication", show=False, dpi=200,
                    save=f"{output_dir}/example_plot_placebo.png")
print(f"Saved: example_plot_placebo.png")

# 图4: 综合诊断图
print("Generating plot 4: diagnostics...")
fig4 = plot_diagnostics(check1,
                        title="Diagnostic Panel",
                        style="publication", show=False, dpi=200,
                        save=f"{output_dir}/example_plot_diagnostics.png")
print(f"Saved: example_plot_diagnostics.png")

# ============================================================
# 示例2: SA设计 - 处理模式热图
# ============================================================
print("\nGenerating SA design example...")
np.random.seed(2025)
n_sa = 80
sa_data = []
for i in range(n_sa):
    if i < 20:
        treat_time = 4
    elif i < 40:
        treat_time = 5
    elif i < 60:
        treat_time = 6
    else:
        treat_time = 99  # never treated
    for t in range(1, 9):
        treat = 1 if t >= treat_time else 0
        y = 5 + 0.3*t + (2.0 if treat else 0) + np.random.normal(0, 0.5)
        sa_data.append({"unit": i, "time": t, "treat": treat, "y": y})

sa_df = pd.DataFrame(sa_data)

# plot_pattern needs a DidCheckResult with SA design
print("Running did_check() for SA design...")
sa_check = did_check(sa_df, formula="y ~ treat", time="time", unit_id="unit",
                     design="sa", n_boot=50, random_seed=42)

# 图5: SA处理模式热图
print("Generating plot 5: SA pattern...")
fig5 = plot_pattern(sa_check,
                    title="Staggered Adoption Treatment Pattern",
                    style="publication", show=False, dpi=200,
                    save=f"{output_dir}/example_plot_pattern.png")
print(f"Saved: example_plot_pattern.png")

print("\n=== All 5 example plots generated successfully! ===")
print(f"\nGenerated files:")
print(f"  1. {output_dir}/example_plot_estimates.png")
print(f"  2. {output_dir}/example_plot_trends.png")
print(f"  3. {output_dir}/example_plot_placebo.png")
print(f"  4. {output_dir}/example_plot_diagnostics.png")
print(f"  5. {output_dir}/example_plot_pattern.png")
