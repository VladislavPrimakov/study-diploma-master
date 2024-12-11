from matplotlib import pyplot as plt
import os, torch


def comapre_all(metric, dct, graph_name, file_name):
    for x in dct:
        metrics1 = torch.load(os.path.join("models", f"{x["file_name"]}.metrics"))
        metric1 = metrics1[metric]
        plt.plot(range(1, len(metric1) + 1), metric1, label=f"{x["model_name"]}", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(graph_name)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("graphs", f"{file_name}.png"))
    plt.clf()

# dct = [
#     {"file_name": "ConvolutionalNetwork_MNIST", "model_name": "Згорткова НМ"},
#     {"file_name": "SiameseNetwork_Conv_Sigma_MNIST", "model_name": "Сіамська НМ із сигмоїдальною функцією активації"},
#     {"file_name": "SiameseNetwork_Conv_Contrastive_MNIST", "model_name": "Сіамська НМ з контрастною функцією втрат"},
#     {"file_name": "SiameseNetwork_Conv_Online_Contrastive_Hard_Negative_MNIST", "model_name": "Сіамська НМ з онлайн контрастною функцією втрат"},
#     {"file_name": "SiameseNetwork_Conv_Triplet_MNIST", "model_name": "Сіамська НМ з триплетною функцією втрат"},
#     {"file_name": "SiameseNetwork_Conv_Online_Triplet_Hard_Negative_MNIST", "model_name": "Сіамська НМ з онлайн триплетною функцією втрат"},
# ]
# comapre_all("test_accuracies", dct, "Порівняння всіх моделей")

# dct = [
#     {"file_name": "SiameseNetwork_Conv_Contrastive_MNIST", "model_name": "НМ з контрастною функцією втрат"},
#     {"file_name": "SiameseNetwork_Conv_Contrastive_GA_MNIST", "model_name": "НМ з контрастною функцією втрат та адаптивним відступом"},
#     {"file_name": "SiameseNetwork_Conv_Online_Contrastive_Hard_Negative_MNIST", "model_name": "НМ з онлайн контрастною функцією втрат Batch Hard"},
# ]
# comapre_all("test_accuracies", dct, "Порівняння моделей з контрастними функціями втрат", "Compare_Contrastive")

# dct = [
#     {"file_name": "SiameseNetwork_Conv_Triplet_MNIST", "model_name": "НМ з контрастною функцією втрат"},
#     {"file_name": "SiameseNetwork_Conv_Triplet_GA_MNIST", "model_name": "НМ з контрастною функцією втрат та адаптивним відступом"},
#     {"file_name": "SiameseNetwork_Conv_Online_Triplet_Hard_Negative_MNIST", "model_name": "НМ з онлайн контрастною функцією втрат Batch Hard"},
# ]
# comapre_all("test_accuracies", dct, "Порівняння моделей з триплетними функціями втрат", "Compare_Triplet")
