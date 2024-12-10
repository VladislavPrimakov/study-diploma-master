import argparse, torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from datasets import DatasetMNISTConvolutional
from networks import ConvolutionalNet
from utils import fit, TypeModel
from losses import AccuracyConv


def main():
    parser = argparse.ArgumentParser(description="Convolutional Network")
    parser.add_argument("--train-batch-size", type=int, default=64, metavar="B", help="Batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="B", help="Batch size for testing (default: 64)")
    parser.add_argument("--epochs", type=int, default=20, metavar="E", help="Number of epochs to train (default: 20)")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="G", help="Learning rate step gamma (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)")
    args = parser.parse_args()

    train_dataset = DatasetMNISTConvolutional("datasets", "train", download=True)
    test_dataset = DatasetMNISTConvolutional("datasets", "test", download=True)

    model_name = "ConvolutionalNetwork_MNIST"
    model = ConvolutionalNet(1, 10)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)

    fit(train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        train_type_model=TypeModel.Conv,
        test_type_model=TypeModel.Conv,
        optimizer=optimizer,
        train_loss_fn=loss_fn,
        test_loss_fn=loss_fn,
        scheduler=scheduler,
        seed=args.seed,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        model_name=model_name,
        train_accuracy_fn=AccuracyConv,
        test_accuracy_fn=AccuracyConv)


if __name__ == "__main__":
    main()
