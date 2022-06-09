from __future__ import print_function, division

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
from recycling_dataset import RecyclingDataset



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # specify which GPU(s) to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STORE_DIR = "/cvhci/temp/p22g5/"
HOME_DIR = "/home/p22g5/AnomaliesRecycling/"
cudnn.benchmark = True


class TransferLearningTrainer:
    """
    Class for training classification model with Transfer Learning
    """

    MODELS_DIR = STORE_DIR + "models/"  # model save directory
    TB_DIR = STORE_DIR + "runs/"  # tensorboard save directory
    DATA_DIR = STORE_DIR + "data/"  # input data directory

    def __init__(self, data_transforms=None, batch_size=32, shuffle=True, num_workers=4):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        if not data_transforms:
            self.data_transforms = self._create_data_transforms_default()
        else:
            self.data_transforms = data_transforms
        self._create_recycling_image_datasets()
        self._create_dataloaders()

    @staticmethod
    def _create_data_transforms_default(self):
        """
        Create default data transformation if no given in constructor
        :param self:
        :return:
        """

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms

    def _create_image_datasets(self):
        """
        Create image datasets
        :return:
        """
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.DATA_DIR, x),
                                                  self.data_transforms[x])
                      for x in ['train', 'test']}
        self.class_names = self.image_datasets['train'].classes

    def _create_recycling_image_datasets(self):
        self.image_datasets = {x: RecyclingDataset(os.path.join(HOME_DIR, "data", x + ".csv"),
                                                       self.data_transforms[x])
                               for x in ['train', 'test']}
        self.class_names = self.image_datasets['train'].classes

    def _create_dataloaders(self):
        """
        Create data loaders
        :return:
        """
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x],
                                                           batch_size=self.batch_size,
                                                           shuffle=self.shuffle,
                                                           num_workers=self.num_workers)
                      for x in ['train', 'test']}

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs=25, model_name=None, early_stop=True):
        """
        Start model training
        :param model: model to train
        :param criterion: criterion
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param num_epochs: number of epochs
        :param model_name: name of the model to save
        :param early_stop: if early stopping should be forced
        :return: trained model with the best accuracy on the test
        """
        if early_stop:
            trigger_times = 0
            patience = scheduler.step_size + 1
            last_loss = 100

        writer = SummaryWriter(self.TB_DIR + model_name)

        model.to(DEVICE)

        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'test']}
        y_loss = {}  # loss history
        y_loss['train'] = []
        y_loss['test'] = []
        y_acc = {}
        y_acc['train'] = []
        y_acc['test'] = []
        if not model_name:
            model_name = model.__class__.__name__
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                y_loss[phase].append(epoch_loss)
                y_acc[phase].append(epoch_acc)

                # deep copy the model
                if phase == 'test':
                    logger_losses = {"test_loss": y_loss['test'][epoch], "train_loss": y_loss['train'][epoch]}
                    logger_acc = {"test_acc": y_acc['test'][epoch], "train_loss": y_acc['train'][epoch]}
                    writer.add_scalars("losses", logger_losses, global_step=epoch)
                    writer.add_scalars("accuracy", logger_acc, global_step=epoch)

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Early Stopping
                if early_stop and phase == 'test':
                    if epoch_loss > last_loss:
                        trigger_times += 1
                        print('Trigger Times:', trigger_times)

                        if trigger_times >= patience:
                            print('Early stopping!')
                            break

                    else:
                        print('trigger times: 0')
                        trigger_times = 0

                last_loss = epoch_loss

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        path = self.MODELS_DIR + model_name + ".pth"
        torch.save(model.state_dict(), path)
        writer.close()
        return model

    @staticmethod
    def imshow(inp, title=None):
        """Imshow for Tensor."""
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
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['test'], 0):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted for {labels[j]}: {self.class_names[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def print_misclassified(self, model_ft):
        """
        Print a list of images that are misclassified in test
        :param model_ft: model used for test
        :return:
        """
        class_names = self.image_datasets['train'].classes
        model_ft.to(DEVICE)
        model_ft.eval()
        with torch.no_grad():
            for i, (input, label) in enumerate(self.image_datasets['test'], 0):
                # print(input)
                input = input.to(DEVICE)
                # label = label.to(device)

                output = model_ft(input.unsqueeze(0))
                _, pred = torch.max(output, 1)

                if label != pred:
                    print(self.image_datasets['test'].imgs[i])
                    # print(dataloaders['test'].dataset.samples[i*batch_size+j][0])
                    print(f'must be {label}, but predicted {class_names[pred]}')

    def print_confusion_matrix(self, model_ft):
        """
        Print confusion matrix with accuracy for each class.
        Takes single image.
        :param model_ft: model used for test
        :return:
        """
        #model_ft.load_state_dict(torch.load("./models/" + model_name))

        n_classes = len(self.class_names)

        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        import pandas as pd

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in self.dataloaders['test']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            output = model_ft(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        # constant for classes

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        matrix_new = [0] * len(cf_matrix)
        for i, row in enumerate(cf_matrix):
            b = [e / sum(row) for e in row]
            matrix_new[i] = b.copy()
        matrix_new = np.array(matrix_new)
        df_cm = pd.DataFrame(matrix_new, index=[i for i in self.class_names],
                             columns=[i for i in self.class_names])
        plt.figure(figsize=(4, 4), dpi=140)
        ax = sn.heatmap(df_cm, fmt=".0%", annot=True, cmap='Blues', cbar=False)
        for i, t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + '(' + str(cf_matrix[i // n_classes, i % n_classes]) + ')')
        plt.yticks(rotation=0)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xticks(rotation=90)
        plt.plot()
        # plt.savefig('output.png')

    def print_missclassified_batches(self, model_ft):
        """
        Print a list of images that are misclassified in test
        Takes batch of images and error-prone.
        :param model_ft: model used for test
        :return:
        """
        batch_size = 4
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['test'], 0):
                # print(labels)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    # print(j)
                    if labels[j] != preds[j]:
                        print(self.dataloaders['test'].dataset.samples[i * batch_size + j][0])
                        print(f'must be {labels[j]}, but predicted {self.class_names[preds[j]]}')

