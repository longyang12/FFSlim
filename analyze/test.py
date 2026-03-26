import matplotlib.pyplot as plt

# --- ATC 风格全局设置 ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

data = {
    "Media 0": [(71, 24), (38258, 23), (54957, 22), (60954, 21), (78189, 20), (80206, 19), (91367, 18), (113201, 17), (113443, 16), (133433, 15), (141708, 14), (151017, 13), (159909, 12), (181611, 11), (183478, 10), (200636, 9), (201885, 24), (224056, 23), (241047, 22), (242511, 21), (261249, 20), (277682, 19), (281086, 18), (293781, 17), (303567, 16)],
    "Media 1": [(37, 11), (10432, 10), (12957, 9), (53489, 11), (84654, 10), (174645, 11), (175739, 10), (188214, 9), (197058, 11), (201739, 10), (243127, 11), (264445, 10)],
    "Media 2": [(20, 4), (73507, 4), (182980, 4), (249425, 4), (310868, 4)]
}

fig, ax = plt.subplots(figsize=(7, 4)) # 紧凑型比例，适合论文插图

colors = ['#1f77b4', '#d62728', '#2ca02c']
markers = ['o', 's', '^']

for i, (key, shifts) in enumerate(data.items()):
    shifts.sort(key=lambda x: x[0]) 
    x = [s[0] for s in shifts]
    y = [s[1] for s in shifts]
    
    ax.plot(x, y, label=key, color=colors[i], marker=markers[i], 
            linewidth=1.5, markersize=5, markerfacecolor='none', markeredgewidth=1)

# 装饰润色
ax.set_xlabel('Access Sequence Index')
ax.set_ylabel('Importance Value')
ax.set_title('Importance Value Shifts and Forgetting')

# --- 关键修复行 ---
# 设置科学计数法，避免横坐标数字太长相互重叠
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# 调整科学计数法那个 "1e5" 的字体大小
ax.xaxis.get_offset_text().set_fontsize(11) 

# 学术去框风格
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.7)

# 图例位置
ax.legend(frameon=True, loc='upper right', edgecolor='none')

plt.tight_layout()
plt.savefig("atc_figure_v2.png", dpi=300)
plt.show()