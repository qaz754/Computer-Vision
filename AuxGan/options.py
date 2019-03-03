import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=50, help='total number of training episodes')
        self.parser.add_argument('--show_every', type=int, nargs='?', default=1000, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=500, help='How often to print scores')
        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='Prints the model being used')
        self.parser.add_argument('--batch', type=int, nargs='?', default=64, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0005, help='learning rate')
        self.parser.add_argument('--beta1', type=int, nargs='?', default=0.5, help='values for beta1')
        self.parser.add_argument('--beta2', type=int, nargs='?', default=0.999, help='values for beta2')

        self.parser.add_argument('--lrelu_val', type=int, nargs='?', default=0.01, help='leaky Relu Value')

        #AUXGan options
        self.parser.add_argument('--num_classes', type=int, nargs='?', default=10, help='WGAN Gradient Penalty value')

        #Discriminator Options
        self.parser.add_argument('--D_hidden', type=int, nargs='+', default=[256, 256], help='hidden layer configuration in a list form for D')
        self.parser.add_argument('--D_activation', type=str, nargs='?', default='lrelu', help='Activation function for the discriminator')
        self.parser.add_argument('--D_input_size', type=int, nargs='?', default=784, help='size of input for the discriminator')
        self.parser.add_argument('--D_output_size', type=int, nargs='?', default=1, help='size of output for the discriminator')

        #Generator Options
        self.parser.add_argument('--G_hidden', type=int, nargs='+', default=[1024, 1024], help='hidden layer configuration in a list form for G')
        self.parser.add_argument('--G_activation', type=str, nargs='?', default='relu', help='Activation function for the generator')
        self.parser.add_argument('--noise_dim', type=int, nargs='?', default=96, help='size of noise input for the generator')
        self.parser.add_argument('--G_output_size', type=int, nargs='?', default=784, help='size of output for the discriminator')
        self.parser.add_argument('--G_out_activation', type=str, nargs='?', default='tanh', help='final output activator')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""
options = options()
opts = options.parse()
batch = opts.batch
"""