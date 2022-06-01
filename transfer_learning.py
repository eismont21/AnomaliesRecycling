from __future__ import print_function, division

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
from torch.autograd import Variable

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # specify which GPU(s) to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(4)
#cudnn.benchmark = True




def _create_data_transforms_default():

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomSolarize(threshold=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=50, p=0.5),

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


def create_image_datasets(data_transforms=None, data_dir='data'):
    if not data_transforms:
        data_transforms = _create_data_transforms_default()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                  for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    return image_datasets, class_names


def create_dataloaders(image_datasets, batch_size=4, shuffle=True, num_workers=4):
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=shuffle, num_workers=num_workers)
                  for x in ['train', 'test']}
    return dataloaders


def train_model(model, criterion, optimizer, scheduler, dataloaders, image_datasets, num_epochs=25, model_name=None):
    trigger_times = 0
    patience = 10
    last_loss = 100

    writer = SummaryWriter('runs/' + model_name)

    model.to(DEVICE)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
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
            for inputs, labels in dataloaders[phase]:
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
            if phase == 'test':
                if epoch_loss > last_loss:
                    trigger_times += 1
                    print('Trigger Times:', trigger_times)

                    if trigger_times >= patience:
                        print('Early stopping!\nStart to test process.')
                        return model

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
    path = "./models/" + model_name
    torch.save(model.state_dict(), path)
    writer.close()
    return model


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


def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test'], 0):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted for {labels[j]}: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def print_missclassified(model_ft, image_datasets):
    class_names = image_datasets['train'].classes
    batch_size = 4
    model_ft.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(image_datasets['test'], 0):
            # print(input)
            input = input.to(DEVICE)
            # label = label.to(device)

            output = model_ft(input.unsqueeze(0))
            _, pred = torch.max(output, 1)

            if label != pred:
                print(image_datasets['test'].imgs[i])
                # print(dataloaders['test'].dataset.samples[i*batch_size+j][0])
                print(f'must be {label}, but predicted {class_names[pred]}')

def print_confusion_matrix(model_ft, dataloaders, class_names):
    #model_ft.load_state_dict(torch.load("./models/" + model_name))

    n_classes = len(class_names)

    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in dataloaders['test']:
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
    df_cm = pd.DataFrame(matrix_new, index=[i for i in class_names],
                         columns=[i for i in class_names])
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

def print_missclassified_batches(model_ft, dataloaders, class_names):
    batch_size = 4
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test'], 0):
            # print(labels)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                # print(j)
                if labels[j] != preds[j]:
                    print(dataloaders['test'].dataset.samples[i * batch_size + j][0])
                    print(f'must be {labels[j]}, but predicted {class_names[preds[j]]}')

