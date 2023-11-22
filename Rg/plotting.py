import matplotlib.pyplot as plt
import numpy as np
import csv

font_title = {"family": 'Times New Roman',
              "style": 'italic',
              "weight": "bold",
              "color": "black",
              "size": 24}
font_label = {"family": 'Times New Roman',
              # "style":'italic',
              "weight": "bold",
              "color": "black",
              "size": 18}
font_legend = {"family": 'Times New Roman',
               # "style":'italic',
               "weight": "bold",
               # "color":"black",
               "size": 16}


def plotting_multiple(steps, epsilon, rg):
    epsilon = np.array(epsilon)
    mean_epsilon = np.mean(epsilon, axis=1)
    std_epsilon = np.std(epsilon, axis=1)

    rg = np.array(rg)
    mean_rg = np.mean(rg, axis=1)
    std_rg = np.std(rg, axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(20, 12))

    ax[0].plot(steps, mean_epsilon, "#544f4f", linewidth=3)
    # ax[0].fill_between(steps, mean_epsilon - std_epsilon,
    #                    mean_epsilon + std_epsilon, color="#caebff", alpha=0.2)

    ax[1].plot(steps, mean_rg, "#fdb296", linewidth=3)
    # ax[1].fill_between(steps, mean_rg - std_rg,
    #                    mean_rg + std_rg, color="#fffade", alpha=0.2)

    ax[0].set_ylabel("Epsilon (Kcal/mol)", font_label)
    ax[1].set_ylabel(r"Predicted Radius of Gyration ($\AA$)", font_label)
    ax[1].set_xlabel("Steps", font_label)
    ax[0].set_xticks([])
    ax[1].set_xticks(steps[::2000])

    labels = ax[1].get_xticklabels() + ax[0].get_yticklabels() + ax[1].get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    [label.set_fontweight("bold") for label in labels]
    [label.set_fontsize(16) for label in labels]

    for i in range(2):
        ax[i].spines["bottom"].set_linewidth(2)
        ax[i].spines["top"].set_linewidth(2)
        ax[i].spines["right"].set_linewidth(2)
        ax[i].spines["left"].set_linewidth(2)

    plt.savefig("reweighting_result.png", dpi=150)
    plt.show()


def plotting_single(steps, epsilon, rg):
    fig = plt.figure(figsize=(20, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(steps, epsilon, "#fdb296", linewidth=3, label='Epsilon')
    ax2 = ax.twinx()
    ax2.plot(steps, rg, 'green', linewidth=3, label="Target")
    ax.legend(loc="center right", prop=font_legend, bbox_to_anchor=(1.003, 0.61), frameon=False)
    ax2.legend(loc="center right", prop=font_legend, bbox_to_anchor=(0.995, 0.5), frameon=False)
    ax.grid(False)
    ax.set_xlabel("Steps", font_label)
    ax.set_ylabel("Epsilon (Kcal/mol)", font_label)
    ax2.set_ylabel(r"Predicted Radius of Gyration ($\AA$)", font_label)
    # plt.tick_params(labelsize=16)
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    [label.set_fontweight("bold") for label in labels]
    [label.set_fontsize(16) for label in labels]
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    plt.savefig("reweighting_result.png", dpi=600)


def read_data(filename):
    with open(filename, 'r') as f:
        table = csv.reader(f)

        next(table)
        steps, epsilons, rgs = [], [], []

        for row in table:
            steps.append(row[0])
            epsilons.append(float(row[1]))
            rgs.append(float(row[2]))

    return steps, np.array(epsilons), np.array(rgs)


if __name__ == '__main__':
    steps, eps, rgs = read_data('result_1.csv')
    steps_, eps_, rgs_ = read_data('result_2.csv')

    step_all = steps if len(steps_) >= len(steps) else steps_
    eps_all = np.stack([eps[:len(step_all)], eps_[:len(step_all)]], axis=1)
    rgs_all = np.stack([rgs[:len(step_all)], rgs_[:len(step_all)]], axis=1)

    plotting_multiple(step_all, eps_all, rgs_all)

