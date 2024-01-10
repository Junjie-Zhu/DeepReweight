import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data from csv files
# optimizers = ['Adam', 'Adadelta', 'RMSprop', 'SGD', 'ASGD', 'Rprop']
optimizers = ['Adam', 'Adadelta', 'Rprop']

results = []
for items in optimizers:
    df = pd.read_csv('%s.csv' % items.lower())
    rgs = df['rg'].tolist()

    if len(rgs) >= 2000:
        results.extend(rgs[:2000])
    else:
        results.extend(rgs + [np.nan] * (2000 - len(rgs)))

results = np.array(results).reshape(-1, 2000)

# choose colors from Spectral
# cmap = plt.get_cmap('viridis')
# colors = [cmap(i) for i in np.linspace(0, 1, len(results))]
colors = ['#fabc74', '#e8505b', '#34a7b2', '#34a7b2', '#e8505b', '#544f4f',]

# Plotting
fig, ax = plt.subplots(figsize=(14, 5))

for i in range(len(results)):
    ax.plot(results[i], label=optimizers[i], color=colors[i], linewidth=2, alpha=0.7)

# get the last non-nan element index in results[0]
non_nan_index_0 = len(results[0]) - np.isnan(results[0]).sum() - 1
non_nan_index_1 = len(results[1]) - np.isnan(results[1]).sum() - 1

ax.hlines(results[0][non_nan_index_0], non_nan_index_0, 2000,
          color=colors[0], linewidth=2, alpha=0.7, linestyle='--')
ax.hlines(results[1][non_nan_index_1], non_nan_index_1, 2000,
          color=colors[1], linewidth=2, alpha=0.7, linestyle='--')
ax.hlines(33, 0, 2000, color='#544f4f', linewidth=2, alpha=0.7, linestyle='--', label='Exp.')

# Set font properties
font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 28}

ax.set_xlabel('Steps', font_properties)
ax.set_ylabel(r'Radius of Gyration ($\AA$)', font_properties)

ax.tick_params(axis='both', labelsize=font_properties['size'],
               labelfontfamily=font_properties['family'])

# ax.set_yticks([10, 12, 14])
# plt.ylim(10, 14)

ax.set_xticks([0, 500, 1000, 1500, 2000])
plt.xlim(0, 2000)

# Set axis width
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

# plt.legend(prop=font_properties, frameon=False)
plt.tight_layout()
plt.savefig('optimizers_ncbd.png', dpi=300)
plt.show()
