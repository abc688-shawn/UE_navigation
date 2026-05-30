#!/usr/bin/env python3
"""
plot_stereo_accuracy.py
读取 result/depth_test_log.csv（由 stereo_accuracy_test.py 生成），
绘制双目测距精度分析图。
输出：result/depth_test_analysis.png（同时弹出交互窗口）

用法：
    python scripts/plot_stereo_accuracy.py
"""

import csv
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────────────
_DIR      = os.path.dirname(__file__)
CSV_PATH  = os.path.join(_DIR, '..', 'result', 'depth_test_log.csv')
OUT_PATH  = os.path.join(_DIR, '..', 'result', 'depth_test_analysis.png')

# ── 读取数据 ─────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV_PATH)))

x_cm       = np.array([float(r['x_cm'])        for r in rows])
true_dist  = np.array([float(r['true_dist_m'])  for r in rows])
meas_depth = np.array([float(r['meas_depth_m']) for r in rows])
error      = np.array([float(r['error_m']) if r['error_m'] != 'N/A' else np.nan
                       for r in rows])
valid_r    = np.array([float(r['valid_ratio'])  for r in rows])
in_range   = np.array([r['in_range'] == '1'     for r in rows])
time_s     = np.array([float(r['time_s'])        for r in rows])

# ── 全局样式 ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      ['Heiti TC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans'],
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'figure.facecolor': '#1c1c2e',
    'axes.facecolor':   '#252540',
    'axes.edgecolor':   '#555580',
    'axes.labelcolor':  '#ccccdd',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'grid.color':       '#3a3a5a',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'text.color':       '#ddddee',
    'legend.facecolor': '#2e2e4a',
    'legend.edgecolor': '#555580',
})

BLUE   = '#4fc3f7'
ORANGE = '#ffb74d'
RED    = '#ef5350'
GREEN  = '#66bb6a'
GRAY   = '#78909c'
YELLOW = '#fff176'

# ── 预计算统计量 ──────────────────────────────────────────────────────────────
err_abs    = np.abs(error)
err_mean   = np.nanmean(error)
err_std    = np.nanstd(error)
err_in     = error[in_range]
err_in_abs = np.abs(err_in)

# ── 布局：2×2 子图 ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.suptitle('双目测距精度分析  |  路线 (−1000, 0) → (500, 0) cm，墙位于 x = 500 cm',
             fontsize=15, fontweight='bold', color='#eeeeff', y=0.97)

gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                      left=0.07, right=0.97, top=0.91, bottom=0.07)

ax1 = fig.add_subplot(gs[0, :])   # 顶部宽图：真实距离 vs 测量深度
ax2 = fig.add_subplot(gs[1, 0])   # 左下：误差随真实距离变化
ax3 = fig.add_subplot(gs[1, 1])   # 右下：真实 vs 测量散点图


# ══════════════════════════════════════════════════════════════════════════════
# 图 1：真实距离 vs SGBM 测量深度（以鱼的 X 坐标为横轴）
# ══════════════════════════════════════════════════════════════════════════════
ax = ax1

# 超出可靠范围区域（灰色底）
ax.axvspan(x_cm[0], x_cm[~in_range].max() if (~in_range).any() else x_cm[0],
           color='#333355', alpha=0.5, zorder=0)
# 进入测距范围区域（深绿色底）
if in_range.any():
    ax.axvspan(x_cm[in_range].min(), x_cm.max(),
               color='#1b3a1b', alpha=0.7, zorder=0)

ax.plot(x_cm, true_dist,  color=BLUE,   lw=2.0, label='真实剩余距离（到墙）', zorder=3)
ax.plot(x_cm, meas_depth, color=ORANGE, lw=1.8, alpha=0.9,
        label='SGBM 测量深度（中央区域中位数）', zorder=3)

ax.fill_between(x_cm, true_dist, meas_depth,
                where=(meas_depth > true_dist),
                color=RED, alpha=0.18, label='高估区域（测量 > 真实）', zorder=2)
ax.fill_between(x_cm, true_dist, meas_depth,
                where=(meas_depth < true_dist),
                color=GREEN, alpha=0.25, label='低估区域（测量 < 真实）', zorder=2)

ax.axvline(x=x_cm[in_range][0] if in_range.any() else 300,
           color=GREEN, lw=1.5, ls=':', alpha=0.9)
ax.text(x_cm[in_range][0] + 10 if in_range.any() else 310, 0.8,
        '← 进入测距\n   可靠范围',
        color=GREEN, fontsize=9, va='bottom')

ax.set_xlabel('鱼的 X 坐标 (cm)')
ax.set_ylabel('距离 / 深度 (m)')
ax.set_title('① 真实剩余距离  vs  SGBM 测量深度')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True)
ax.set_xlim(x_cm[0], x_cm[-1])
ax.set_ylim(0, max(true_dist.max(), meas_depth.max()) * 1.08)

ax.text(x_cm[0] + 30, true_dist.max() * 0.82,
        '超出双目可靠范围\n（基线 3cm，理论上限 ~2m）',
        color='#8888aa', fontsize=9)


# ══════════════════════════════════════════════════════════════════════════════
# 图 2：误差随真实距离变化（核心诊断图）
# ══════════════════════════════════════════════════════════════════════════════
ax = ax2

# 零线与 ±0.5m 参考带
ax.axhline(0, color=GRAY, lw=1.2, ls='--', alpha=0.7)
ax.axhspan(-0.5, 0.5, color='#2a4a2a', alpha=0.4, label='误差 ±0.5m 参考带')

mask_out = ~in_range
mask_in  =  in_range

ax.scatter(true_dist[mask_out], error[mask_out],
           c=GRAY,  s=18, alpha=0.55, label='超出测距范围', zorder=3)
ax.scatter(true_dist[mask_in],  error[mask_in],
           c=GREEN, s=40, alpha=0.9,  label='在测距范围内（<2m）', zorder=4,
           edgecolors='#aaffaa', linewidths=0.5)

valid_mask = ~np.isnan(error)
z = np.polyfit(true_dist[valid_mask], error[valid_mask], 1)
x_fit = np.linspace(true_dist.min(), true_dist.max(), 200)
ax.plot(x_fit, np.polyval(z, x_fit),
        color=YELLOW, lw=1.5, ls='--', alpha=0.8, label=f'线性趋势  斜率={z[0]:+.3f}')

stats_text = (f'全程误差统计\n'
              f'均值:  {err_mean:+.3f} m\n'
              f'标准差: {err_std:.3f} m\n'
              f'───────────\n'
              f'测距范围内（<2m）\n'
              f'均值:  {np.nanmean(err_in):+.3f} m\n'
              f'标准差: {np.nanstd(err_in):.3f} m')
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1e3a',
                  edgecolor='#555580', alpha=0.9))

ax.set_xlabel('真实剩余距离 (m)')
ax.set_ylabel('误差 = 测量 − 真实 (m)')
ax.set_title('② 误差随真实距离的分布')
ax.legend(loc='upper left', fontsize=8.5)
ax.grid(True)
ax.invert_xaxis()   # 从远到近（与鱼的行驶方向一致）


# ══════════════════════════════════════════════════════════════════════════════
# 图 3：真实距离 vs 测量深度散点 + 理想线
# ══════════════════════════════════════════════════════════════════════════════
ax = ax3

# 理想线（y = x）
lim_max = max(true_dist.max(), meas_depth.max()) * 1.05
ax.plot([0, lim_max], [0, lim_max],
        color=GRAY, lw=1.5, ls='--', alpha=0.7, label='理想（测量 = 真实）')
ax.fill_between([0, lim_max], [0 - 0.5, lim_max - 0.5], [0 + 0.5, lim_max + 0.5],
                color='#2a4a2a', alpha=0.35, label='±0.5m 误差带')

ax.scatter(true_dist[mask_out], meas_depth[mask_out],
           c=GRAY,  s=16, alpha=0.5, label='超出测距范围')
ax.scatter(true_dist[mask_in],  meas_depth[mask_in],
           c=GREEN, s=45, alpha=0.9, label='在测距范围内',
           edgecolors='#aaffaa', linewidths=0.5)

z2 = np.polyfit(true_dist, meas_depth, 1)
x_fit2 = np.linspace(true_dist.min(), true_dist.max(), 200)
ax.plot(x_fit2, np.polyval(z2, x_fit2),
        color=ORANGE, lw=1.8, ls='-.',
        label=f'线性拟合  y={z2[0]:.3f}x{z2[1]:+.3f}')

ss_res = np.sum((meas_depth - np.polyval(z2, true_dist)) ** 2)
ss_tot = np.sum((meas_depth - meas_depth.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
        fontsize=10, va='top', color=ORANGE,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e1e3a',
                  edgecolor='#555580', alpha=0.9))

ax.set_xlabel('真实剩余距离 (m)')
ax.set_ylabel('SGBM 测量深度 (m)')
ax.set_title('③ 真实距离 vs 测量深度（散点图）')
ax.legend(loc='lower right', fontsize=8.5)
ax.grid(True)
ax.set_xlim(0, lim_max)
ax.set_ylim(0, lim_max)
ax.set_aspect('equal', adjustable='box')


# ── 保存 & 显示 ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'[plot] 已保存 → {OUT_PATH}')
plt.show()
