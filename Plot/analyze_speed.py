import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load data from csv
df = pd.read_csv('time.csv')

# Get data as list
mc = df['mc'].tolist()[:3]
deep = df['deep'].tolist()

# load step data
df_step = pd.read_csv('step.csv', encoding='gbk')
steps = np.sort(df_step['steps'].tolist())

# transform steps into 4 catalogs by step number
small, medium, large, huge = 0, 0, 0, 0
for items in steps:
    if items <= 100:
        small += 1
    elif 100 < items <= 1000:
        medium += 1
    elif 1000 < items <= 2000:
        large += 1
    else:
        huge += 1

# make unpaired non-parametric test
# statistic, p = stats.mannwhitneyu(mc, deep)

# make unpaires parametric test
# statistic, p = stats.ttest_ind(mc, deep)
# print(p)

# log 10 transform
# mc = [np.log10(i * 1000) for i in mc]
# deep = [np.log10(i * 1000) for i in deep]

# get viridis cmap and extract colors
cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, len(steps))]

# Plotting
fig, ax = plt.subplots(figsize=(9, 8))

# Set font properties
font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 30}

ax.bar(1, small, color='#544f4f', alpha=0.7, width=0.3,
       edgecolor='black', linewidth=1.5,)
ax.bar(1.5, medium, color='#34a7b2', alpha=0.7, width=0.3,
       edgecolor='black', linewidth=1.5,)
ax.bar(2, large, color='#fabc74', alpha=0.7, width=0.3,
       edgecolor='black', linewidth=1.5,)
ax.bar(2.5, huge, color='#e8505b', alpha=0.7, width=0.3,
       edgecolor='black', linewidth=1.5,)

# Draw a bar plot to show the difference, adding error bars
# ax.bar(1, np.mean(mc), yerr=np.std(mc), color='#e8505b', alpha=0.7,
#        width=0.12, capsize=5,
#        edgecolor='black', linewidth=1.5,)
# ax.bar(1.3, np.mean(deep), yerr=np.std(deep), color='#34a7b2', alpha=0.7,
#        width=0.12, capsize=5,
#        edgecolor='black', linewidth=1.5,)

# Set xticks and their range
ax.set_xticks([1, 1.5, 2, 2.5])
ax.set_xticklabels(['<100', '100-1000', '1000-2000', '>2000'])

# ax.set_xticks([1, 1.3])
# ax.set_xticklabels(['MC Reweighting', 'DeepReweighting'])
# plt.xlim(0.8, 1.5)

# Set yticks and their range
# ax.set_yticks(np.arange(0, 15, 3))
# plt.ylim(0, 15)

# Apply to all ticks
ax.tick_params(axis='both', labelsize=26,
               labelfontfamily=font_properties['family'])

# Set axis width
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

# Set labels
ax.set_xlabel('Optimize Step Required', font_properties)
ax.set_ylabel('System Number', font_properties)

# ax.set_xlabel('Method', font_properties)
# ax.set_ylabel('Time (s)', font_properties)

# plt.tight_layout()
plt.savefig('step.png', dpi=300)
plt.show()
