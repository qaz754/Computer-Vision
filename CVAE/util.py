
import matplotlib.pyplot as plt
import numpy as np
import torch


import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def confusion_plot(matrix, y_category):
    '''
    A function that plots a confusion matrix
    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def accuracy(net, loader):
    '''
    A function that returns total number of correct predictions and total comparisons
    given a neural net and a pytorch data loader

    :param net: neural net
    :param loader: data loader
    :return:
    '''

    correct = 0
    total = 0

    for images, labels in iter(loader):

        output = net.forward(images)

        _, prediction = torch.max(output.data, 1)

        total += labels.shape[0] #accumulate by batch_size
        correct += (prediction == labels).sum() #accumulate by total_correct

    return correct, total

def prediction_accuracy(net, images, labels):
    '''
    A function that returns total number of correct predictions and total comparisons
    given a neural net and a pytorch data loader

    :param net: neural net
    :param loader: data loader
    :return:
    '''

    output = net.forward(images)

    _, prediction = torch.max(output.data, 1)

    total = labels.shape[0] #accumulate by batch_size
    correct = (prediction == labels).sum() #accumulate by total_correct

    return correct, total


def pred_plotter(original_image, prediction, y_label):
    '''
    Used for visual inspection of how well the classifier works on an image.

    Takes in the original_image, softmax class predictions for the image, and y_labels to plot a side by side graph that
    shows what the model predicted, and what the model looks like.

    :param original_image (tensor): a tensor that holds the values pixel values for the image
    :param prediction (tensor): a tensor that holds the softmax class probabilities for original_image
    :param y_label (list): a list that holds names for the classes.
    :return:
    '''

    fig, (ax1, ax2) = plt.subplots(figsize=(9,6), ncols=2)

    y_values = np.arange(len(y_label))

    ax1.barh(y_values, prediction.squeeze().numpy(), align = 'center')
    ax1.set_yticks(y_values)
    ax1.set_yticklabels(y_label) #use the name of the classes as labels
    ax1.invert_yaxis()

    #values for the axis and the title
    ax1.set_xlabel('Probability')
    ax1.set_title('Class Probability')

    #shows the original_image
    ax2.imshow(original_image.view(1, 28, 28).squeeze().numpy())

    plt.show()

def show_images(images, filename, iterations, title=None):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        '''global title'''
        if title == None:
            title = 'LS-CGANs After %s iterations'
        plt.suptitle(title %iterations)
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
        plt.savefig(filename)
