from __future__ import print_function, division

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchmetrics
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
import pandas as pd
from datetime import datetime
import json
from torch.autograd import Variable
from image_classification.recycling_dataset import RecyclingDataset
import warnings

warnings.filterwarnings("ignore")
from image_classification.stratified_batch import StratifiedBatchSampler
from image_classification.decision_accuracy import DecisionAccuracy
from image_classification.constants import Constants

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # specify which GPU(s) to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


class PolysecureClassifier:
    """
    Class for training classification model with transfer learning and make predictions.
    """

    def __init__(
        self,
        data_transforms=None,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        sos="",
        save_results=False,
        config=None,
        synthetic=True,
    ):
        self.sos = (sos, "")
        self.synthetic = synthetic
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.save_results = save_results
        self.result = {}
        if self.save_results:
            self._create_store_folder()
            self.result.update(config)
            with open(self.store_dir + "config.json", "w") as f:
                json.dump(config, f)
        else:
            self.store_dir = Constants.STORE_DIR.value
        if data_transforms is None:
            self._create_data_transforms_default()
        else:
            self.data_transforms = data_transforms
        self._create_recycling_image_datasets()
        self._create_dataloaders()

    def _create_data_transforms_default(self):
        """
        Create default data transformation if no given in constructor
        """
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        self.data_transforms = data_transforms

    def _create_recycling_image_datasets(self):
        """
        Create train and test datasets
        """
        self.image_datasets = {
            x: RecyclingDataset(
                os.path.join(
                    Constants.PROJECT_DIR.value, "data", self.sos[i], x + ".csv"
                ),
                Constants.DATA_DIR.value,
                self.data_transforms[x],
                sos=self.sos[0],
                synthetic=self.synthetic,
            )
            for i, x in enumerate(["train", "test"])
        }
        self.class_names = self.image_datasets["train"].classes

    def _create_dataloaders(self):
        """
        Create data loaders
        """
        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.image_datasets[x],
                batch_sampler=StratifiedBatchSampler(
                    self.image_datasets[x].img_labels["count"],
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                ),
                # batch_size=self.batch_size,
                # shuffle=self.shuffle,
                num_workers=self.num_workers,
            )
            for i, x in enumerate(["train", "test"])
        }

    def _create_store_folder(self):
        """
        Create store folder for storing experiment results
        """
        time = datetime.now()
        folder_name = time.strftime("%Y-%m-%d_%H-%M-%S/")
        self.result["timestamp"] = time
        path = Constants.STORE_DIR.value + "experiments/" + folder_name
        os.makedirs(path)
        os.makedirs(path + "model")
        os.makedirs(path + "images")
        os.makedirs(path + "results")
        self.store_dir = path

    def train_model(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs=12,
        model_name=None,
        early_stopping=True,
    ):
        """
        Start model training
        :param model: model to train
        :param criterion: loss function
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param num_epochs: number of epochs
        :param model_name: name of the model to save
        :param early_stopping: if True, early stop the model if no improvement in validation loss
        :return: trained model
        """
        if not model_name:
            model_name = model.__class__.__name__
        model.to(DEVICE)
        if early_stopping:
            trigger_times = 0
            patience = scheduler.step_size + 1
            last_loss = 100

        dataset_sizes = {x: len(self.image_datasets[x]) for x in ["train", "test"]}
        writer = SummaryWriter(self.store_dir + "tensorboard")
        y_loss = {"train": [], "test": []}  # loss history
        y_acc = {"train": [], "test": []}  # accuracy history

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc, best_r2, best_de_acc = 0.0, 0.0, 0.0
        best_mae, best_mse = 1.0, 1.0

        since = time.time()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                get_mae = torchmetrics.MeanAbsoluteError().to(DEVICE)
                get_mse = torchmetrics.MeanSquaredError().to(DEVICE)
                get_r2 = torchmetrics.R2Score().to(DEVICE)
                get_de_acc = DecisionAccuracy().to(DEVICE)

                # Iterate over data
                for sample in self.dataloaders[phase]:
                    inputs = sample["image"]
                    labels = sample["label"]
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        get_mae(preds, labels)
                        get_mse(preds, labels)
                        get_r2(preds, labels)
                        get_de_acc(preds, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_mae = get_mae.compute()
                epoch_mse = get_mse.compute()
                epoch_r2 = get_r2.compute()
                epoch_de_acc = get_de_acc.compute()

                print(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
                    f"De_Acc : {epoch_de_acc:.4f} MAE: {epoch_mae:.4f} "
                    f"MSE: {epoch_mse:.4f} R^2: {epoch_r2:.4f}"
                )

                get_mae.reset()
                get_mse.reset()
                get_r2.reset()
                get_de_acc.reset()
                y_loss[phase].append(epoch_loss)
                y_acc[phase].append(epoch_acc)

                # deep copy the model
                if phase == "test":
                    logger_losses = {
                        "test_loss": y_loss["test"][epoch],
                        "train_loss": y_loss["train"][epoch],
                    }
                    logger_acc = {
                        "test_acc": y_acc["test"][epoch],
                        "train_loss": y_acc["train"][epoch],
                    }
                    writer.add_scalars("losses", logger_losses, global_step=epoch)
                    writer.add_scalars("accuracy", logger_acc, global_step=epoch)

                # deep copy the model
                if phase == "test":
                    if (
                        (epoch_de_acc > best_de_acc)
                        or (epoch_de_acc == best_de_acc)
                        and (best_acc < epoch_acc)
                    ):
                        best_de_acc = epoch_de_acc
                        best_acc = epoch_acc
                        best_mae = epoch_mae
                        best_mse = epoch_mse
                        best_r2 = epoch_r2
                        best_model_wts = copy.deepcopy(model.state_dict())

                # Early Stopping
                if early_stopping and phase == "test":
                    if epoch_loss > last_loss:
                        trigger_times += 1
                        print("Trigger Times:", trigger_times)

                        if trigger_times >= patience:
                            print("Early stopping!")
                            break

                    else:
                        print("trigger times: 0")
                        trigger_times = 0

                last_loss = epoch_loss

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best test acc: {best_acc:4f}")
        print(f"Best test de_acc: {best_de_acc:4f}")
        print(f"Best test mae: {best_mae:4f}")
        print(f"Best test mse: {best_mse:4f}")
        print(f"Best test R^2: {best_r2:4f}")

        self.metrics = {
            "acc": best_acc.item(),
            "de_acc": best_de_acc.item(),
            "mae": best_mae.item(),
            "mse": best_mse.item(),
            "r^2": best_r2.item(),
        }
        self.result.update(self.metrics)

        if self.save_results:
            df_metrics = pd.DataFrame(self.metrics, index=[0])
            df_metrics.to_csv(self.store_dir + "results/metrics.csv", index=False)

        # save best model
        model.load_state_dict(best_model_wts)
        path = self.store_dir + "model/" + model_name + "_weights.pth"
        torch.save(model.state_dict(), path)
        path = self.store_dir + "model/" + model_name + "_model.pth"
        torch.save(model, path)
        writer.close()

        return model

    @staticmethod
    def imshow(inp, title=None):
        """
        Imshow for Tensor.
        :param inp: input image tensor
        :param title: title of the image
        """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def visualize_model(self, model, num_images=6):
        """
        Visualize the model predictions.
        :param model: trained model to visualize
        :param num_images: number of images to visualize
        """
        was_training = model.training
        model.eval()
        images_so_far = 0
        _ = plt.figure()

        with torch.no_grad():
            for i, sample in enumerate(self.dataloaders["test"], 0):
                inputs, labels = sample["image"], sample["label"]
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis("off")
                    ax.set_title(
                        f"Predicted for {labels[j]}: {self.class_names[preds[j]]}"
                    )
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def print_misclassified(self, model_ft, dataset="test", plot=False):
        """
        Print and visualize a list of images that are misclassified by the model
        :param model_ft: trained model
        :param dataset: test or train dataset
        :param plot: whether to plot the images
        """
        class_names = self.image_datasets["train"].classes
        model_ft.to(DEVICE)
        model_ft.eval()
        images = {}
        with torch.no_grad():
            for i, sample in enumerate(self.image_datasets[dataset], 0):
                input, label = sample["image"], sample["label"]
                input = input.to(DEVICE)

                output = model_ft(input.unsqueeze(0))
                _, pred = torch.max(output, 1)

                if label != pred:
                    title = self.image_datasets[dataset][i]["img_path"][
                        self.image_datasets[dataset][i]["img_path"].find("data") + 5 :
                    ]
                    images[title] = (label, class_names[pred])
                    title += (
                        "\n" + f"must be {label}, but predicted {class_names[pred]}"
                    )
                    if plot:
                        _ = plt.figure(figsize=(4, 4), dpi=140)
                        ax = plt.subplot()
                        ax.set_title(title)
                        inp = input.cpu().data
                        inp = inp.numpy().transpose((1, 2, 0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        inp = std * inp + mean
                        inp = np.clip(inp, 0, 1)
                        ax.imshow(inp)
                        ax.axis("off")
                        plt.pause(0.001)
                    elif not self.save_results:
                        print(title)

        self.misclassified_images = images

        if self.save_results:
            names, counts, preds = [], [], []
            for image, (label, pred) in images.items():
                names.append(image)
                counts.append(label)
                preds.append(pred)

            df_images = pd.DataFrame({"name": names, "count": counts, "pred": preds})
            df_images.to_csv(
                self.store_dir + "results/misclassified_images.csv", index=False
            )

    def print_confusion_matrix(self, model_ft):
        """
        Print confusion matrix with accuracy for each class.
        :param model_ft: trained model
        """
        model_ft.to(DEVICE)
        model_ft.eval()

        from sklearn.metrics import confusion_matrix
        import seaborn as sn

        y_pred = []
        y_true = []

        # iterate over test data
        for sample in self.dataloaders["test"]:
            inputs, labels = sample["image"], sample["label"]
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            output = model_ft(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        # Build confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        self.confusion_diagonal = []
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = "%.0f%%\n%d/%d" % (p, c, s)
                    self.confusion_diagonal.append(p / 100)
                elif c == 0:
                    annot[i, j] = ""
                else:
                    annot[i, j] = "%.0f%%\n%d" % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=140)
        sn.heatmap(cm, cmap="Blues", annot=annot, fmt="", ax=ax, cbar=False)
        plt.yticks(rotation=0)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        plt.xticks(rotation=90)
        plt.plot()

        if self.save_results:
            plt.savefig(self.store_dir + "images/" + "confusion_matrix.png")
            plt.close("all")
            plt.clf()
            plt.show()

            for i, x in enumerate(self.confusion_diagonal):
                self.result[str(i)] = x
            df_confusion_diagonal = pd.DataFrame(
                self.confusion_diagonal,
                columns=range(len(self.confusion_diagonal)),
                index=[0],
            )
            df_confusion_diagonal.to_csv(
                self.store_dir + "results/confusion_diagonal.csv", index=False
            )
        else:
            plt.show()

    def update_results(self):
        """
        Update results.csv with current experiment results.
        """
        df_results = pd.read_csv(Constants.PROJECT_DIR.value + "data/results.csv")
        df_results = pd.concat(
            [df_results, pd.DataFrame.from_records([self.result])], ignore_index=True
        )
        df_results.to_csv(Constants.PROJECT_DIR.value + "data/results.csv", index=False)
