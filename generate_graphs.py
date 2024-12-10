from matplotlib import pyplot as plt
import os, torch


def compare_two(metric, file_name1, file_name2, model_name1, model_name2, graph_name):
    metrics1 = torch.load(os.path.join("models", f"{file_name1}.metrics"))
    metric1 = metrics1[metric]
    metrics2 = torch.load(os.path.join("models", f"{file_name2}.metrics"))
    metric2 = metrics2[metric]

    plt.plot(range(1, len(metric1) + 1), metric1, label=f"{model_name1}", marker="o")
    plt.plot(range(1, len(metric2) + 1), metric2, label=f"{model_name2}", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(graph_name)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("graphs", f"Compare_{file_name1}_{file_name2}.png"))
    plt.clf()


def comapre_all(metric, dct, graph_name):
    for x in dct:
        metrics1 = torch.load(os.path.join("models", f"{x["file_name"]}.metrics"))
        metric1 = metrics1[metric]
        plt.plot(range(1, len(metric1) + 1), metric1, label=f"{x["model_name"]}", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(graph_name)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("graphs", f"Compare_all.png"))
    plt.clf()

# compare_two("test_accuracies","SiameseNetwork_Conv_Contrastive_MNIST", "SiameseNetwork_Conv_Online_Contrastive_Hard_Negative_MNIST",
#             "Сіамська НМ з контрастною функцією втрат", "Сіамська НМ з онлайн контрастною функцією втрат",
#             "Порівняння сіамських мереж з контрастною функцією втрат")

# compare_two("test_accuracies","SiameseNetwork_Conv_Triplet_MNIST", "SiameseNetwork__Conv_Online_Triplet_Hard_Negative_MNIST",
#             "Сіамська НМ з триплетною функцією втрат", "Сіамська НМ з онлайн триплетною функцією втрат",
#             "Порівняння сіамських мереж з триплетною функцією втрат")

dct = [
    {"file_name": "ConvolutionalNetwork_MNIST", "model_name": "Згорткова НМ"},
    {"file_name": "SiameseNetwork_Conv_Sigma_MNIST", "model_name": "Сіамська НМ із сигмоїдальною функцією активації"},
    {"file_name": "SiameseNetwork_Conv_Contrastive_MNIST", "model_name": "Сіамська НМ з контрастною функцією втрат"},
    {"file_name": "SiameseNetwork_Conv_Online_Contrastive_Hard_Negative_MNIST", "model_name": "Сіамська НМ з онлайн контрастною функцією втрат"},
    {"file_name": "SiameseNetwork_Conv_Triplet_MNIST", "model_name": "Сіамська НМ з триплетною функцією втрат"},
    {"file_name": "SiameseNetwork_Conv_Online_Triplet_Hard_Negative_MNIST", "model_name": "Сіамська НМ з онлайн триплетною функцією втрат"},
]

comapre_all("test_accuracies", dct, "Порівняння всіх моделей")