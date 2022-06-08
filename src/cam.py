import torch
from torchray.attribution.grad_cam import grad_cam
import matplotlib.pyplot as plt
import os


class CAM:
    """
    Class Activation Map
    Better understanding of the results
    """
    def __init__(self, trainer):
        """
        Constructor.
        :param trainer: trainer that contains image_datasets['test'] and class_names
        """
        self.trainer = trainer

    def print_cam(self, model_ft):
        """
        Prints Class Activation Map
        :param model_ft:
        :return:
        """
        import random
        i = random.randint(0, len(self.trainer.image_datasets['test']))
        print(self.trainer.image_datasets['test'].imgs[i][0])
        model = model_ft
        x = self.trainer.image_datasets['test'][i][0].unsqueeze(0)
        label = self.trainer.image_datasets['test'][i][1]
        # Grad-CAM backprop.
        input = self.trainer.image_datasets['test'][i][0]
        output = model_ft(input.unsqueeze(0))
        _, pred = torch.max(output, 1)
        # print("Predicted", pred.detach().numpy()[0])
        prediction = pred.detach().numpy()[0]
        saliency = grad_cam(model, x, label, saliency_layer='layer4.1.conv2', resize=True)
        self.plot_example_custom(x, saliency, 'grad-cam backprop', label, prediction, self.trainer.class_names)

    @staticmethod
    def plot_example_custom(input,
                            saliency,
                            method,
                            category_id,
                            prediction,
                            class_names,
                            show_plot=False,
                            save_path=None):
        """
        Plot an example. Custom version of the :class:`torchray.benchmark.plot_example`

        :param input: 4D tensor containing input images.
        :param saliency: 4D tensor containing saliency maps.
        :param method: name of saliency method.
        :param category_id: ID of category.
        :param prediction: prediction of the model.
        :param class_names: all class names.
        :param show_plot: f True, show plot. Default: ``False``.
        :param save_path: Path to save figure to. Default: ``None``.
        :return: shows heatmap
        """
        from torchray.utils import imsc

        if isinstance(category_id, int):
            category_id = [category_id]

        batch_size = len(input)

        plt.clf()
        for i in range(batch_size):
            class_i = category_id[i % len(category_id)]

            plt.subplot(batch_size, 2, 1 + 2 * i)
            imsc(input[i])
            plt.title('input image', fontsize=8)

            plt.subplot(batch_size, 2, 2 + 2 * i)
            imsc(saliency[i], interpolation='none')
            plt.title('{} for label {} (predicted {})'.format(
                method, class_names[class_i], prediction), fontsize=8)

        # Save figure if path is specified.
        if save_path:
            save_dir = os.path.dirname(os.path.abspath(save_path))
            # Create directory if necessary.
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ext = os.path.splitext(save_path)[1].strip('.')
            plt.savefig(save_path, format=ext, bbox_inches='tight')

        # Show plot if desired.
        if show_plot:
            plt.show()
