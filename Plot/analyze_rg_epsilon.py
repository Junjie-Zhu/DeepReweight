import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel file
file_path = 'rg-mjx.csv'
df = pd.read_csv(file_path, encoding='gbk')

# Read data as lists
protein_names = df['system'].tolist()
# protein_names[8] = 'E3 ubiquitin\n ligase RNF4'
# protein_names[9] = 'N-term VS\n Virus phosphoprotein'

exp_rg = df['exp'].tolist()
sim_rg = df['sim'].tolist()

mc_rg = df['mc_pred'].tolist()
mc_epsilon = df['mc_epsilon'].tolist()

rg = df['pred'].tolist()
epsilon = df['epsilon'].tolist()

rprop_rg = df['pred_Rprop'].tolist()
rprop_epsilon = df['epsilon_Rprop'].tolist()

mc_new_rg = df['mc_ntop_pred'].tolist()
mc_new_epsilon = df['mc_ntop_epsilon'].tolist()

# get mean and std from mc
mc_all_rg = np.array([mc_rg, mc_new_rg]).T
mc_all_epsilon = np.array([mc_epsilon, mc_new_epsilon]).T

mc_rg_mean = np.mean(mc_all_rg, axis=1)
mc_epsilon_mean = np.mean(mc_all_epsilon, axis=1)

mc_rg_std = np.std(mc_all_rg, axis=1)
mc_epsilon_std = np.std(mc_all_epsilon, axis=1)

# get mean and std from rprop
rprop_all_rg = np.array([rprop_rg, rg]).T
rprop_all_epsilon = np.array([rprop_epsilon, epsilon]).T

rprop_rg_mean = np.mean(rprop_all_rg, axis=1)
rprop_epsilon_mean = np.mean(rprop_all_epsilon, axis=1)

rprop_rg_std = np.std(rprop_all_rg, axis=1)
rprop_epsilon_std = np.std(rprop_all_epsilon, axis=1)

# choose colors from Spectral
cmap = plt.get_cmap('Spectral')
colors = [cmap(i) for i in np.linspace(0, 1, 4)]

# Plotting
fig, ax = plt.subplots(figsize=(25, 8))

ax.scatter(protein_names, mc_new_epsilon, color='#e8505b', s=10, marker='o', label='MC Reweighting')
ax.plot(protein_names, mc_new_epsilon, color='#e8505b', linewidth=2, alpha=0.7)
ax.hlines(np.mean(mc_new_epsilon), 0, 27, color='#e8505b', linewidth=1, alpha=0.7, linestyle='--')

ax.scatter(protein_names, rprop_epsilon, color='#34a7b2', s=10, marker='o', label='DeepReweighting')
ax.plot(protein_names, rprop_epsilon, color='#34a7b2', linewidth=2, alpha=0.7)
ax.hlines(np.mean(rprop_epsilon), 0, 27, color='#34a7b2', linewidth=1, alpha=0.7, linestyle='--')

ax.hlines(0.164306, 0, 27, color='#544f4f', linewidth=1, alpha=0.7, linestyle='--', label='Simulation')

# ax.scatter(protein_names, exp_rg, color='#544f4f', s=10, marker='o', label='Experiment')
# ax.plot(protein_names, exp_rg, color='#544f4f', linewidth=2, alpha=0.7, linestyle='--')
#
# ax.scatter(protein_names, sim_rg, color='#fabc74', s=10, marker='o', label='Simulation')
# ax.plot(protein_names, sim_rg, color='#fabc74', linewidth=2, alpha=0.7, linestyle='--')
#
# ax.scatter(protein_names, mc_new_rg, color='#e8505b', s=10, marker='o', label='MC Reweighting')
# ax.plot(protein_names, mc_new_rg, color='#e8505b', linewidth=2, alpha=0.7)
#
# ax.scatter(protein_names, rprop_rg, color='#34a7b2', s=10, marker='o', label='DeepReweighting')
# ax.plot(protein_names, rprop_rg, color='#34a7b2', linewidth=2, alpha=0.7)

# # Draw the mean lines and std area, adding scatters for means
# ax.scatter(protein_names, mc_rg_mean, color='#e8505b', s=10, marker='o', label='MC Reweighting')
# ax.plot(protein_names, mc_rg_mean, color='#e8505b', linewidth=2, alpha=0.7)
# ax.fill_between(protein_names, mc_rg_mean - mc_rg_std,
#                 mc_rg_mean + mc_rg_std, color="#e8505b", alpha=0.3)
#
# ax.scatter(protein_names, rprop_rg_mean, color='#34a7b2', s=10, marker='o', label='DeepReweighting')
# ax.plot(protein_names, rprop_rg_mean, color='#34a7b2', linewidth=2, alpha=0.7)
# ax.fill_between(protein_names, rprop_rg_mean - rprop_rg_std,
#                 rprop_rg_mean + rprop_rg_std, color="#34a7b2", alpha=0.3)

# Set the x ticks to rotate 70 degrees
ax.set_xticks(protein_names)
ax.set_xticklabels(protein_names, rotation=70, ha='right')

# Set font properties
font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 24}

# Apply to all ticks
ax.tick_params(axis='both', labelsize=font_properties['size'],
               labelfontfamily=font_properties['family'])

# Set axis width
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

# Apply legend and use font properties
plt.legend(prop=font_properties, frameon=False, loc='upper left')

# Set labels
ax.set_ylabel("Epsilon (Kcal/mol)", font_properties)
# ax.set_ylabel(r"Radius of Gyration ($\AA$)", font_properties)

# make sure all x-ticks will be saved in figure area
plt.gcf().subplots_adjust(top=2, bottom=1)
# fig.autofmt_xdate()

# plt.tight_layout()
plt.savefig('axis.png', dpi=300)
plt.show()
