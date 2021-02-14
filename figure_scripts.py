import matplotlib.pyplot as plt


def plot_curves(train, val, f_name):
    plt.clf()
    plt.plot(train, label="Train Acc")
    plt.plot(list(range(0, len(val) * 100, 100)), val, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/" + f_name + ".png")
    plt.close()
