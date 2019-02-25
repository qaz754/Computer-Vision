
from model import *
from image_to_gif import *

#using pytorch's dataloader
from data_loader import *
from image_folder import *
from torch.utils.data import DataLoader
import util

from options import options

from collections import deque

options = options()
opts = options.parse()

tf.reset_default_graph()

# number of images for each batch
batch_size = opts.batch
# our noise dimension

session = tf.InteractiveSession()

# placeholder for images from the training dataset
x = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.channel_in])

label = tf.placeholder(tf.float32, [None, opts.num_classes])
label_domain = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.num_classes])

target_label = tf.placeholder(tf.float32, [None, opts.num_classes])
target_domain = tf.placeholder(tf.float32, [None, opts.image_shape, opts.image_shape, opts.num_classes])

# Used for adaptive LR decay
adaptive_lr = tf.placeholder(tf.float32, shape=[])

Discrim = PatchDiscrim(32, name='discriminator', opts=opts)
generator = ResNet(opts.channel_in, opts.channel_out, up_channel=opts.channel_up, name='generator')

# random noise fed into our generator
# generated images

with tf.variable_scope("") as scope:
    G_sample = generator.output(tf.concat([x, target_domain],  axis=3))

    scope.reuse_variables()
    recon_cycle = generator.output(tf.concat([G_sample, label_domain],  axis=3))

recon_loss = recon_loss(x, recon_cycle) * opts.cycle_lambda

with tf.variable_scope("") as scope:
    #scale images to be -1 to 1
    logits_real_src, logits_real_cls = Discrim.discriminator(util.preprocess_img(x))

    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake_src, logits_fake_cls = Discrim.discriminator(util.preprocess_img(G_sample))

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

# get our solver
D_solver, G_solver = get_solvers(learning_rate=adaptive_lr)

# get our loss
D_loss_src, G_loss_src = gan_loss(logits_real_src, logits_fake_src)

D_loss_cls = classification_loss(logits_real_cls, label)
G_loss_cls = classification_loss(logits_fake_cls, target_label)

D_loss = tf.reduce_mean(D_loss_src + opts.cls_lambda * D_loss_cls)
G_loss = tf.reduce_mean(G_loss_src + opts.cls_lambda * G_loss_cls)

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

# a giant helper function
def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step, num_epoch=10):
    """Train a GAN for a certain number of epochs.

    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """

    # compute the number of iterations we need

    image_dir = '/home/youngwook/Downloads/img_align_celeba/'
    attribute_file = '/home/youngwook/Downloads/list_attr_celeba.txt'
    folder_names = get_folders(image_dir)

    train_data = Pix2Pix_AB_Dataloader(folder_names[0], attribute_file=attribute_file, size=opts.resize, randomcrop=opts.image_shape)
    train_loader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=12)

    recon = []
    file_list = []
    last_100_loss_dq = deque(maxlen=100)
    last_100_loss = []
    steps = 0

    checkpoint_dir = './model'
    saver = tf.train.Saver()

    if opts.resume:
        #print('Loading Saved Checkpoint')
        load_session(checkpoint_dir, saver, session, model_name=opts.model_name)


    for epoch in range(num_epoch):
        # every show often, show a sample result
        lr = util.linear_LR(epoch, opts)

        for (image, og_label, target_labels) in train_loader:
            #
            label_image = util.expand_spatially(og_label, opts.image_shape)

            target_label_image = util.expand_spatially(target_labels, opts.image_shape)
            #transform the data to be the right shape

            og_label = og_label.float().numpy()
            target_labels = target_labels.float().numpy()
            image = image.float().numpy()

            # run a batch of data through the network
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={x: image, label: og_label, target_domain: target_label_image, adaptive_lr: lr})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={x: image, label: og_label, label_domain: label_image, target_label: target_labels, target_domain: target_label_image, adaptive_lr: lr})

            last_100_loss_dq.append(G_loss_curr)
            last_100_loss.append(np.mean(last_100_loss_dq))

            # print loss every so often.
            # We want to make sure D_loss doesn't go to 0
            if steps % opts.show_every == 0:

                samples = sess.run(G_sample, feed_dict={x: image, target_domain: target_label_image})

                recon_name = './img/recon_%s.png' % steps
                recon.append(recon_name)
                util.show_images(samples[:opts.batch], opts, recon_name)
                util.plt.show()

                file_name = './img/original_%s.png' % steps
                file_list.append(file_name)
                util.show_images(image[:opts.batch], opts, file_name)
                util.plt.show()

            if steps % opts.print_every == 0:
                print('Epoch: {}, D: {:.4}, G:{:.4}'.format(steps, D_loss_curr, G_loss_curr))
                util.raw_score_plotter(last_100_loss)

            if steps % opts.save_every == 0:
                if opts.save_progress:
                    save_session(saver, session, checkpoint_dir, steps, model_name=opts.model_name)

            steps += 1

    util.raw_score_plotter(last_100_loss)
    print('Final images')

    image_to_gif('', recon, duration=1, gifname='recon')
    image_to_gif('', file_list, duration=1, gifname='original')

tf.global_variables_initializer().run()
run_a_gan(session, G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step, num_epoch=opts.epoch)
