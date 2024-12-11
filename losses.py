import torch, random
import torch.nn as nn
import torch.nn.functional as F
from deap import base, creator, tools, algorithms
import numpy as np


class GA:
    def __init__(self, opt, fitness_func, **kwargs):
        low = 0.1
        up = 1
        creator.create("FitnessMax", base.Fitness, weights=(opt,))
        creator.create("Individ", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform, low, up)
        self.toolbox.register("individualCreator", tools.initRepeat, creator.Individ, self.toolbox.attribute, n=1)
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.individualCreator)
        self.toolbox.register("evaluate", fitness_func, **kwargs)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=up, eta=20)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=20, indpb=1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self, iter):
        hof = tools.HallOfFame(1)
        population = self.toolbox.populationCreator(n=20)
        algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.1, ngen=iter, halloffame=hof, verbose=False)
        return hof[0][0]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1, threshold=0.7, useGA=False, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.useGA = useGA
        self.reduction = reduction
        self.eps = 1e-9
        self.thresholds = []
        self.reset_train = False
        self.margin = margin
        self.threshold = threshold

    def forward(self, type_fit, output1, output2, targets):
        distances = torch.norm(output1 - output2, dim=1)
        if type_fit == "train":
            if self.reset_train:
                self.thresholds = []
                self.reset_train = False
            if self.useGA:
                ga = GA(opt=1.0, fitness_func=self._evalute_fitness, distances=distances, targets=targets)
                self.threshold = ga.run(50)
                self.thresholds.append(self.threshold)
        if type_fit == "test":
            if self.useGA and self.reset_train == False:
                self.threshold = np.mean(self.thresholds)
                self.reset_train = True
                print(f"Threshold: {self.threshold:.4f}")
        losses = 0.5 * (targets.float() * distances + (1 - targets).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        loss = losses.mean() if self.reduction == "mean" else losses.sum()
        predict = torch.where(distances < self.threshold, 1, 0)
        accuracy = (predict == targets).sum().item() / len(targets)
        return loss, accuracy

    def _evalute_fitness(self, margin, distances, targets):
        predict = torch.where(distances < margin[0], 1, 0)
        accuracy = (predict == targets).sum().item() / len(targets)
        return accuracy,


class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin=1, threshold=0.7, reduction="mean"):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, embeddings, targets):
        positive_pairs, negative_pairs, accuracy = self._hardNegativePairSelector(embeddings, targets)
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()
        return loss, accuracy

    def _hardNegativePairSelector(self, embeddings, targets):
        distance_matrix = torch.cdist(embeddings, embeddings, p=2).to(embeddings.device)
        batch_size = targets.size(0)
        all_pairs = torch.combinations(torch.arange(batch_size), r=2).to(embeddings.device)
        positive_pairs = all_pairs[targets[all_pairs[:, 0]] == targets[all_pairs[:, 1]]]
        negative_pairs = all_pairs[targets[all_pairs[:, 0]] != targets[all_pairs[:, 1]]]
        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        top_negative_distances, top_indices = torch.topk(negative_distances, k=len(positive_pairs), largest=False)
        top_negative_pairs = negative_pairs[top_indices]

        positive_predictions = (distance_matrix[positive_pairs[:, 0], positive_pairs[:, 1]] < self.threshold)
        accuracy = positive_predictions.sum().float() / len(positive_predictions)
        return positive_pairs, top_negative_pairs, accuracy.to("cpu")


class TripletLoss(nn.Module):
    def __init__(self, margin=1, useGA=False, reduction="mean"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.margins = []
        self.reset_train = False
        self.reduction = reduction
        self.useGA = useGA

    def forward(self, type_fit, anchor, positive, negative):
        distance_ap = (anchor - positive).pow(2).sum(1)
        distance_an = (anchor - negative).pow(2).sum(1)
        distance_difference = distance_ap - distance_an

        if type_fit == "train":
            if self.reset_train:
                self.margins = []
                self.reset_train = False
            if self.useGA:
                ga = GA(opt=-1.0, fitness_func=self._evalute_fitness, distance_difference=distance_difference, reduction=self.reduction)
                self.margin = ga.run(50)
                self.margins.append(self.margin)
        if type_fit == "test":
            if self.useGA and self.reset_train == False:
                self.margin = np.mean(self.margin)
                self.reset_train = True
                print(f"Margin: {self.margin}:.4f")

        losses = F.relu(distance_difference + self.margin)
        loss = losses.mean() if self.reduction == "mean" else losses.sum()
        accuracy = ((distance_ap < distance_an).sum().item() / len(distance_ap))
        return loss, accuracy

    def _evalute_fitness(self, margin, distance_difference, reduction):
        losses = F.relu(distance_difference + margin[0])
        loss = losses.mean() if reduction == "mean" else losses.sum()
        return loss.item(),


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=1, reduction="mean"):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings, targets):
        triplets = self._hardNegativeTripletSelector(embeddings, targets)
        distance_ap = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        distance_an = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(distance_ap - distance_an + self.margin)
        loss = losses.mean() if self.reduction == "mean" else losses.sum()
        accuracy = ((distance_ap < distance_an).sum().item() / len(distance_ap))
        return loss, accuracy

    def _hardNegativeTripletSelector(self, embeddings, targets):
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        triplets = []
        for label in torch.unique(targets):
            label_mask = (targets == label)
            label_indices = torch.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = torch.where(~label_mask)[0]
            anchor_positives = torch.combinations(label_indices, r=2)
            for anchor_positive in anchor_positives:
                ap_distance = distance_matrix[anchor_positive[0], anchor_positive[1]]
                negative_distances = distance_matrix[anchor_positive[0], negative_indices]
                loss_values = ap_distance - negative_distances + self.margin
                hard_negative_idx = torch.argmax(loss_values)
                if loss_values[hard_negative_idx] > 0:
                    hard_negative = negative_indices[hard_negative_idx]
                    triplets.append([anchor_positive[0].item(), anchor_positive[1].item(), hard_negative.item()])
        if len(triplets) == 0 and len(label_indices) > 1 and len(negative_indices) > 0:
            triplets.append([label_indices[0].item(), label_indices[1].item(), negative_indices[0].item()])
        return torch.tensor(triplets, dtype=torch.long)


def AccuracyEmbeddingPair(output1, output2, targets, threshold):
    distances = torch.norm(output1 - output2, dim=1)
    predict = torch.where(distances <= threshold, 1, 0)
    return (predict == targets).sum().item() / len(targets)


def AccuracyEmbeddingTriplet(output1, output2, output3):
    distances_ap = torch.norm(output1 - output2, dim=1)
    distances_an = torch.norm(output1 - output3, dim=1)
    predict = torch.where(distances_ap < distances_an, 1, 0)
    return predict.sum().item() / len(output1)


def AccuracySigma(output, target):
    predict = torch.where(output > 0.5, 1, 0)
    return (predict == target).sum().item() / len(target)


def AccuracyConv(output1, output2, target1, target2):
    predict = (torch.argmax(output1, dim=1) == target1) & (torch.argmax(output2, dim=1) == target2)
    return predict.sum().item() / len(predict)
