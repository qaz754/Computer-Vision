
import util
import numpy as np

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=50):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every

    def train(self):
        steps = 0

        for e in range(self.epochs):
            running_loss = 0

            correct = 0
            total = 0

            for images, labels in iter(self.trainloader):
                steps += 1

                # flatten mnist images into a 784 long vector
                #images.size()[0] is the batch_size

                #images = images.view(images.size()[0], -1)
                #print(images.shape)
                self.optimizer.zero_grad()

                # forward and backward passes

                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                correct_, total_ = util.prediction_accuracy(self.model, images, labels)

                correct += correct_
                total += total_

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "LossL {:.4f}".format(running_loss / self.print_every),
                          "Running Accuracy {:4f}".format(correct.numpy() / np.float(total)))

                    running_loss = 0