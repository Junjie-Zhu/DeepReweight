import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rg_penalty(result_list, reference_list):
    """
    Calculate the penalty of radius of gyration.
    :param result_list: list, predicted radius of gyration
    :param reference_list: list, experimental radius of gyration
    :return: penalty: list
    """
    penalty = []
    for i in range(len(result_list)):
        penalty.append(np.abs(result_list[i] - reference_list[i]) / reference_list[i])

    return penalty


# Load data from csv files
acc_results = pd.read_csv('acc.csv', encoding='gbk')
result_1 = acc_results['1'].tolist()
result_10 = acc_results['10'].tolist()
result_100 = acc_results['100'].tolist()
result_1000 = acc_results['1000'].tolist()
result_10000 = acc_results['10000'].tolist()
result_exp = acc_results['exp'].tolist()
result_mc = acc_results['mc_pred'].tolist()

colors = ['#544f4f', '#f7e294', '#9acca6', '#5a92b6', '#301437', '#a53449']

# Plotting
fig, ax = plt.subplots(figsize=(12, 12))

# make a boxplot from rg_penalty lists, using color from colors

boxes = ax.boxplot([rg_penalty(result_1, result_exp), rg_penalty(result_10, result_exp),
                    rg_penalty(result_100, result_exp), rg_penalty(result_1000, result_exp),
                    rg_penalty(result_10000, result_exp), rg_penalty(result_mc, result_exp)],
                   positions=[1, 1.8, 2.6, 3.4, 4.2, 5.2], widths=0.45, patch_artist=True,
                   showmeans=True, showfliers=True)

i = 0
for box, color in zip(boxes['boxes'], colors):
    box.set_facecolor(color)
    box.set_alpha(0.3)
    # box.set_edgecolor(color)
    box.set_linewidth(0)

    # Set median type
    plt.setp(boxes['medians'][i], color=color, linewidth=2)

    # Set mean type
    mean_marker = 'o'
    plt.setp(boxes['means'][i], marker=mean_marker, markerfacecolor=color, markeredgecolor=color, markersize=8)
    plt.setp(boxes['fliers'][i], marker='o', markerfacecolor=color, markeredgecolor=color, markersize=5)

    # Set whiskers and caps
    plt.setp(boxes['whiskers'][2 * i], color=color, linewidth=2)
    plt.setp(boxes['whiskers'][2 * i + 1], color=color, linewidth=2)
    plt.setp(boxes['caps'][2 * i], color=color, linewidth=2)
    plt.setp(boxes['caps'][2 * i + 1], color=color, linewidth=2)
    i += 1

# Set font properties
font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 28}

ax.set_xlabel('Accuracy', font_properties)
ax.set_ylabel('Rg Penalty', font_properties)

ax.set_xticks([1, 1.8, 2.6, 3.4, 4.2, 5.2])
# set xticklabels to 10^-8, 10^-9, 10^-10, 10^-11, 10^-12
ax.set_xticklabels([r'$10^{-8}$', r'$10^{-9}$', r'$10^{-10}$', r'$10^{-11}$', r'$10^{-12}$', 'MC'])

ax.tick_params(axis='both', labelsize=font_properties['size'],
               labelfontfamily=font_properties['family'])

# Set axis width
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.tight_layout()
plt.savefig('acc.png', dpi=300)
plt.show()
