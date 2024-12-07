import torch, os
from sampler import BalancedBatchSampler
from tqdm import tqdm
from enum import Enum, auto
from matplotlib import pyplot as plt


def fit(train_dataset, test_dataset, model, train_type_model, test_type_model, train_loss_fn, test_loss_fn, optimizer, scheduler, epochs, seed, train_batch_size, test_batch_size,
        model_name, balanced_sampler=False, train_accuracy_fn=None, test_accuracy_fn=None):
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_kwargs = {"batch_size": train_batch_size}
    test_kwargs = {"batch_size": test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    if balanced_sampler:
        train_kwargs.update({"sampler": BalancedBatchSampler(train_dataset, train_dataset.dataset.targets)})

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    metrics = {"train_losses": [], "test_losses": [], "train_accuracies": [], "test_accuracies": []}
    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, train_type_model, device, train_loader, optimizer, train_loss_fn, epoch, train_accuracy_fn)
        test_loss, test_accuracy = test(model, test_type_model, device, test_loader, test_loss_fn, test_accuracy_fn)
        metrics["train_losses"].append(train_loss)
        metrics["train_accuracies"].append(train_accuracy)
        metrics["test_losses"].append(test_loss)
        metrics["test_accuracies"].append(test_accuracy)
        scheduler.step()

    torch.save(model.state_dict(), os.path.join("models", f"{model_name}.model"))
    torch.save(metrics, os.path.join("models", f"{model_name}.metrics"))
    saveGraphs(model_name, epochs, metrics)


class TypeModel(Enum):
    Conv = auto()
    Pairs = auto()
    Triplet = auto()
    Sigma = auto()
    Online = auto()


def loss_model(model, type_model, data, loss_fn, accuracy_fn=None, stage=None):
    match type_model:
        case TypeModel.Conv:
            input1, input2, target1, target2 = data[0], data[1], data[2], data[3]
            output1 = model(input1)
            output2 = model(input2)
            loss = loss_fn(output1, target1)
            accuracy = accuracy_fn(output1, output2, target1, target2)
        case TypeModel.Pairs:
            input1, input2, target = data[0], data[1], data[2]
            output1 = model(input1)
            output2 = model(input2)
            loss = loss_fn(output1, output2, target)
            accuracy = accuracy_fn(output1, output2, target)
        case TypeModel.Triplet:
            input1, input2, input3 = data[0], data[1], data[2]
            output1 = model(input1)
            output2 = model(input2)
            output3 = model(input3)
            loss = loss_fn(output1, output2, output3)
            accuracy = accuracy_fn(output1, output2, output3)
        case TypeModel.Sigma:
            input1, input2, target = data[0], data[1], data[2]
            output = model(input1, input2).squeeze(1)
            loss = loss_fn(output, target)
            accuracy = accuracy_fn(output, target)
        case TypeModel.Online:
            input, target = data[0], data[1]
            output = model(input)
            loss, accuracy = loss_fn(output, target)

    return loss, accuracy


def train(model, type_model, device, train_loader, optimizer, loss_fn, epoch, accuracy_fn=None):
    model.train()
    train_tqdm = tqdm(train_loader, leave=True)
    train_loss = 0
    train_accuracy = 0
    l = len(train_loader)
    for data in train_tqdm:
        data = [x.to(device) for x in data]
        optimizer.zero_grad()
        loss, accuracy = loss_model(model, type_model, data, loss_fn, accuracy_fn=accuracy_fn)
        train_accuracy += accuracy
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_tqdm.set_description(f"[Epoch: {epoch}, Loss: {train_loss / l:.4f}, Accuracy: {train_accuracy / l:.4f}]")
    return train_loss / l, train_accuracy / l


def test(model, type_model, device, test_loader, loss_fn, accuracy_fn=None):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    l = len(test_loader)
    with torch.no_grad():
        for data in test_loader:
            data = [x.to(device) for x in data]
            loss, accuracy = loss_model(model, type_model, data, loss_fn, accuracy_fn=accuracy_fn)
            test_accuracy += accuracy
            test_loss += loss.item()
    test_accuracy = test_accuracy / l
    test_loss = test_loss / l
    print(f"\nTest set: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f})")
    return test_loss, test_accuracy


def saveGraphs(model_name, epochs, metrics):
    plt.plot(range(1, epochs + 1), metrics["train_losses"], label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), metrics["test_losses"], label="Test Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("graphs", f"{model_name}_Loss.png"))
    plt.clf()

    plt.plot(range(1, epochs + 1), metrics["train_accuracies"], label="Train Accuracy", marker="o")
    plt.plot(range(1, epochs + 1), metrics["test_accuracies"], label="Test Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("graphs", f"{model_name}_Accuracy.png"))
    plt.clf()
