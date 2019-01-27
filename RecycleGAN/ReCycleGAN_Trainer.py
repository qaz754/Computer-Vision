import util
import os
import numpy as np
from image_to_gif import image_to_gif
from memory import ReplayBuffer
import torch
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=2500, cycle_lambda=10, pred_lambda=10):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every
        self.cycle_lambda = cycle_lambda
        self.pred_lambda = pred_lambda

    def train(self):
        steps = 0

        last_100_loss = deque(maxlen=100)
        # used to make gifs later
        fpred_image_list = []
        gpred_image_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []

        Input_deque = ReplayBuffer(50, 50, 0)
        Target_deque = ReplayBuffer(50, 50, 0)

        directory = './img/'


        for e in range(self.epochs):

            running_loss = 0

            for input_image, target_image in iter(self.trainloader):

                for i in range(len(input_image)):
                    input_image[i] = input_image[i].to(device)
                    target_image[i] = target_image[i].to(device)
                '''
                Gen Loss
                Gen Input_0
                Gen Input_1
                Gen Input_2
                
                Gen Target_0
                Gen Target_1
                Gen Target_2

                Gen Input_2 Pred
                Gen Target_2 Pred
                                
                Gen Input_2 Cycle (Uses Temporally Predicted Restructured Input_2)
                Gen Target_2 Cycle (Uses Temporally Predicted Restructured Target_2)
                '''

                '''
                Input_Image To Target_Image
                '''

                target_0_gen = self.model[0](input_image[0])
                target_1_gen = self.model[0](input_image[1])
                target_2_gen = self.model[0](input_image[2])

                next_target_pred = self.model[4](torch.cat((target_image[0], target_image[1]), 1))

                Target_deque.add(target_0_gen.detach(), target_1_gen.detach(), target_2_gen.detach(), next_target_pred.detach())

                '''
                Cycle
                (Input0, Input1) => (target0, target1)_gen => predict_target2(cat(target0_gen, target1_gen)) => Input2_Gen 
                '''
                input_2_restruct = self.model[1](self.model[4](torch.cat((target_0_gen, target_1_gen), 1)))

                '''
                Target_Image To Input_Image
                '''
                input_0_gen = self.model[1](target_image[0])
                input_1_gen = self.model[1](target_image[1])
                input_2_gen = self.model[1](target_image[2])

                next_input_pred = self.model[5](torch.cat((input_image[0], input_image[1]), 1))

                Input_deque.add(input_0_gen.detach(), input_1_gen.detach(), input_2_gen.detach(), input_2_restruct.detach())

                '''
                Cycle
                (target0, target1) => (Input0, Input1)_gen => predict_target2(cat(Input0_gen, Input1_gen)) => Target2_Gen 
                '''
                target_2_restruct = self.model[0](self.model[5](torch.cat((input_0_gen, input_1_gen), 1)))


                '''Losses'''
                '''Generator Loss'''
                target0_loss = util.generator_loss(self.model[2](target_0_gen).detach())
                target1_loss = util.generator_loss(self.model[2](target_1_gen).detach())
                target2_loss = util.generator_loss(self.model[2](target_2_gen).detach())

                target_gen_loss = target0_loss + target1_loss + target2_loss

                input0_loss = util.generator_loss(self.model[3](input_0_gen).detach())
                input1_loss = util.generator_loss(self.model[3](input_1_gen).detach())
                input2_loss = util.generator_loss(self.model[3](input_2_gen).detach())

                input_gen_loss = input0_loss + input1_loss + input2_loss

                '''Prediction Loss'''
                next_input_loss  = self.criterion(next_input_pred, input_image[2])
                next_target_loss = self.criterion(next_target_pred, target_image[2])

                pred_loss = next_input_loss + next_target_loss

                '''CycleLoss'''
                input_cycle_loss = self.criterion(input_2_restruct, input_image[2])
                target_cycle_loss = self.criterion(target_2_restruct, target_image[2])

                cycle_loss = input_cycle_loss + target_cycle_loss

                gen_loss = target_gen_loss + input_gen_loss + self.pred_lambda * pred_loss + self.cycle_lambda * cycle_loss


                '''
                Discrim Loss
                
                Input_0
                Input_1
                Input_2
                Future PredInput_2
                
                Target_0
                Target_1
                Target_2
                Future PredTarget_2
                
                '''

                target_0_gen_history, target_1_gen_history, target_2_gen_history, target_2_pred_history = Target_deque.sample()

                target0_D_loss = util.discriminator_loss(self.model[2](target_image[0]), self.model[2](target_0_gen_history))
                target1_D_loss = util.discriminator_loss(self.model[2](target_image[1]), self.model[2](target_1_gen_history))
                target2_D_loss = util.discriminator_loss(self.model[2](target_image[2]), self.model[2](target_2_gen_history))
                target2_D_pred_loss = util.discriminator_loss(self.model[2](target_image[2]), self.model[2](target_2_pred_history))

                target_discrim_loss = target0_D_loss + target1_D_loss + target2_D_loss + target2_D_pred_loss

                input_0_gen_history, input_1_gen_history, input_2_gen_history, input_2_pred_history = Input_deque.sample()

                input0_D_loss = util.discriminator_loss(self.model[3](input_image[0]), self.model[3](input_0_gen_history))
                input1_D_loss = util.discriminator_loss(self.model[3](input_image[1]), self.model[3](input_1_gen_history))
                input2_D_loss = util.discriminator_loss(self.model[3](input_image[2]), self.model[3](input_2_gen_history))
                input2_D_pred_loss = util.discriminator_loss(self.model[3](input_image[2]), self.model[3](input_2_pred_history))

                input_discrim_loss = input0_D_loss + input1_D_loss + input2_D_loss + input2_D_pred_loss


                '''Model Update'''
                self.optimizer[0].zero_grad()
                gen_loss.backward()
                self.optimizer[0].step()

                self.optimizer[1].zero_grad()
                target_discrim_loss.backward()
                self.optimizer[1].step()

                '''G Discriminator'''
                self.optimizer[2].zero_grad()
                input_discrim_loss.backward()
                self.optimizer[2].step()

                last_100_loss.append(gen_loss.item())
                running_loss += gen_loss.item()
                train_loss.append(gen_loss.item())

                if steps % self.print_every == 0:
                    print('\rEpoch {}\tSteps: {}\tLoss: {:.4f}\n'.format(e, steps, np.mean(last_100_loss)), end="")

                    fpred_image_list.append(util.save_images_to_directory(target_0_gen, directory, 'target_gen_%s.png' % steps))
                    gpred_image_list.append(util.save_images_to_directory(input_0_gen, directory, 'input_gen_%s.png' % steps))
                    input_image_list.append(util.save_images_to_directory(input_image[0], directory, 'input_image_%s.png' % steps))
                    target_image_list.append(util.save_images_to_directory(target_image[0], directory, 'target_image_%s.png' % steps))

                    torch.save(self.model[0].state_dict(), './model/Target_Gen_%s.pth' %steps)
                    torch.save(self.model[1].state_dict(), './model/Input_Gen_%s.pth' %steps)
                    torch.save(self.model[2].state_dict(), './model/Target_Discrim_%s.pth' %steps)
                    torch.save(self.model[3].state_dict(), './model/Input_Discrim_%s.pth' %steps)

                steps += 1

        torch.save(self.model[0].state_dict(), './model/Target_Gen_%s.pth' % steps)
        torch.save(self.model[1].state_dict(), './model/Input_Gen_%s.pth' % steps)
        torch.save(self.model[2].state_dict(), './model/Target_Discrim_%s.pth' % steps)
        torch.save(self.model[3].state_dict(), './model/Input_Discrim_%s.pth' % steps)

        image_to_gif('./img/', fpred_image_list, duration=1, gifname='Target_Gen')
        image_to_gif('./img/', gpred_image_list, duration=1, gifname='Input_Gen')
        image_to_gif('./img/', target_image_list, duration=1, gifname='Target_Image')
        image_to_gif('./img/', input_image_list, duration=1, gifname='Input_Image')

        util.raw_score_plotter(train_loss)

