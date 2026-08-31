import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch as tc
import torchvision.transforms.v2 as tvs
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from convnet import ConvNet
from datasets import CatsAndDogs
from scatnet import ScatNet

EPOCHS: int = 10
RNG = tc.Generator().manual_seed(0)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", choices=["convnet", "scatnet"], default="convnet")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    transforms = tvs.Compose([
        tvs.Grayscale(),
        tvs.ToImage(),
        # tvs.CenterCrop(128),
        tvs.Resize((128, 128)),
        tvs.RandomHorizontalFlip(), # bit of data augmentation
        tvs.ConvertImageDtype(tc.float32),
    ])
    dataset = CatsAndDogs(Path(os.environ["CATS_AND_DOGS"]), transform=transforms)

    train_dataset, valid_dataset = tc.utils.data.random_split(dataset, [0.8, 0.2], generator=RNG)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = tc.device("cuda")

    SHAPE = (1, 128, 128)
    match args.model:
        case "convnet":
            model = ConvNet(shape=SHAPE)
        case "scatnet":
            model = ScatNet(shape=SHAPE)
        case other:
            raise ValueError(f"Unknown model {other}")

    model = model.to(device)
    # model.compile()
    criterion = tc.nn.CrossEntropyLoss().to(device)
    optimizer = tc.optim.Adam(model.parameters())

    acc_train = []; f1_train = []; loss_train = []
    acc_valid = []; f1_valid = []; loss_valid = []

    # for epoch in tqdm(range(EPOCHS), "Training"):
    for epoch in range(EPOCHS):
        print("Epoch:", epoch)

        running_loss = []
        running_acc = []
        running_f1 = []
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy, and F1-score
            predicted = tc.argmax(outputs, dim=-1)
            running_loss.append(loss.item())
            running_acc.append(accuracy_score(labels.cpu(), predicted.cpu()))
            running_f1.append(f1_score(labels.cpu(), predicted.cpu(), average="weighted"))
            # print("Batch accuracy score", accuracy_score(labels.cpu(), predicted.cpu()))

        loss_train.append(np.mean(running_loss))
        acc_train.append(np.mean(running_acc))
        f1_train.append(np.mean(running_f1))

        print(
            f"  Train - Loss: {loss_train[epoch]:.4f}, Acc: {acc_train[epoch]:.4f}, F1: {f1_train[epoch]:.4f}"
        )

        # Validation phase
        model.eval()
        running_loss = []
        running_acc = []
        running_f1 = []
        with tc.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = tc.max(outputs, 1)
                running_loss.append(loss.item())
                running_acc.append(accuracy_score(labels.cpu(), predicted.cpu()))
                running_f1.append(
                    f1_score(labels.cpu(), predicted.cpu(), average="weighted")
                )

            loss_valid.append(np.mean(running_loss))
            acc_valid.append(np.mean(running_acc))
            f1_valid.append(np.mean(running_f1))

            print(
                f"  Valid - Loss: {loss_valid[epoch]:.4f}, Acc: {acc_valid[epoch]:.4f}, F1: {f1_valid[epoch]:.4f}"
            )

    tc.save(model.state_dict(), Path("./weights", f"{model.name}.pt"))


if __name__ == "__main__":
    main()
