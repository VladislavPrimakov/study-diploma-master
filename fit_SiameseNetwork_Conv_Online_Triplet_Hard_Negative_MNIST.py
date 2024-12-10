import argparse

import torch.nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from datasets import DatasetMNISTBase, DatasetMNISTTriplet
from networks import EmbeddingNet
from utils import fit, TypeModel
from losses import OnlineTripletLoss, AccuracyEmbeddingTriplet


def main():
    parser = argparse.ArgumentParser(description="Siamese Network with Triplet based on ConvNet")
    parser.add_argument("--train-batch-size", type=int, default=64, metavar="B", help="Batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="B", help="Batch size for testing (default: 64)")
    parser.add_argument("--epochs", type=int, default=20, metavar="E", help="Number of epochs to train (default: 20)")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="G", help="Learning rate step gamma (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)")
    args = parser.parse_args()

    train_dataset = DatasetMNISTBase("datasets", "train", download=True)
    test_dataset = DatasetMNISTTriplet("datasets", "test", download=True)

    model_name = "SiameseNetwork_Conv_Online_Triplet_Hard_Negative_MNIST"
    model = EmbeddingNet(1)
    train_loss_fn = OnlineTripletLoss(margin=1, reduction="sum")
    test_loss_fn = torch.nn.TripletMarginWithDistanceLoss(margin=1, reduction="sum")
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)

    fit(train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        train_type_model=TypeModel.Online,
        test_type_model=TypeModel.Triplet,
        optimizer=optimizer,
        train_loss_fn=train_loss_fn,
        test_loss_fn=test_loss_fn,
        scheduler=scheduler,
        seed=args.seed,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        model_name=model_name,
        balanced_sampler=True,
        test_accuracy_fn=AccuracyEmbeddingTriplet)


if __name__ == "__main__":
    main()