
import util
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tester():

    def __init__(self, opts, trainloader, model):

        self.opts = opts
        self.trainloader = trainloader
        self.model = model

    def test(self):
        steps = 0

        correct = 0
        total = 0

        for images, labels in iter(self.trainloader):
            steps += 1

            confusion = torch.zeros(self.opts.num_classes, self.opts.num_classes)

            # label to be fed into confusion matrix plot.
            y_label = np.arange(self.opts.num_classes)

            output = self.model(images.to(device))

            pred = torch.max(output, 1)[1]

            for pred, label in zip(pred, labels):
                confusion[pred.cpu().item()][label.item()] += 1
                if pred.cpu().item() == label.item():
                    correct += 1
                total += 1

        for i in range(self.opts.num_classes):
            confusion[i] = confusion[i] / confusion[i].sum()

        util.confusion_plot(confusion, list(y_label))

        print("Test Accuracy", correct / float(total))

