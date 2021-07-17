import time
from ops import *
from utils import *
import tensorflow as tf
import numpy as np
import sys
class ResNet(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset

        if self.dataset_name == 'cifar10' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar10()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_cifar100()
            self.img_size = 32
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_mnist()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'fashion-mnist' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_fashion()
            self.img_size = 28
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'tiny' :
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiny()
            self.img_size = 64
            self.c_dim = 3
            self.label_dim = 200


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = len(self.train_x) // self.batch_size
        self.acc = args.acc
        self.init_lr = args.lr


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            ag = []
            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, ag=ag,kernel=3, stride=1, scope='conv')
            print(x.shape)
            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, ag=ag, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, ag=ag,is_training=is_training, downsample=True, scope='resblock1_0')
            print(x.shape)
            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, ag=ag,is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, ag=ag,is_training=is_training, downsample=True, scope='resblock2_0')
            print(x.shape)
            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4,ag=ag, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8,ag=ag, is_training=is_training, downsample=True, scope='resblock_3_0')
            print(x.shape)
            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8,ag=ag, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################
            

            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)
            # ag.append(x)
            x = global_avg_pooling(x)
            x = fully_conneted(x, units=self.label_dim, scope='logit')
            # ag.append(x)
            return x,ag

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.test_inptus = tf.placeholder(tf.float32, [100, self.img_size, self.img_size, self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [100, self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits,self.ag = self.network(self.train_inptus)
        self.test_logits,_ = self.network(self.test_inptus, is_training=False, reuse=True)

        
        self.cross_entropy, self.train_loss, self.train_accuracy,_,_ = classification_loss(logit=self.train_logits, label=self.train_labels)
        _,self.test_loss, self.test_accuracy,self.predictied,self.actual = classification_loss(logit=self.test_logits, label=self.test_labels)
        self.ag_list = []
        for ag in self.ag:
            self.ag_list.append(tf.gradients(self.cross_entropy,ag))
            
        reg_loss = tf.losses.get_regularization_loss()
        self.train_loss += reg_loss
        self.test_loss += reg_loss


        """ Training """
        self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_accuracy = tf.summary.scalar("test_accuracy", self.test_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=self.epoch)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
       #  lr = 1e-3
    #if epoch > 180:
     #   lr *= 0.5e-3
    #elif epoch > 160:
     #   lr *= 1e-3
    #elif epoch > 120:
     #   lr *= 1e-2
    #elif epoch > 80:
     #   lr *= 1e-1
        if could_load:
            #epoch_lr = self.init_lr
            epoch_lr = 1e-3
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
              epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
              epoch_lr = epoch_lr * 0.1

            print(" [*] Load SUCCESS")
        else:
            #epoch_lr = self.init_lr
            epoch_lr = 1e-3
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        
        for epoch in range(start_epoch, self.epoch):
            if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
              epoch_lr = epoch_lr * 0.1
#            if epoch > 180:
#              epoch_lr *= 0.5e-3
#            elif epoch > 160:
#              epoch_lr *= 1e-3
#            elif epoch > 120:
#              epoch_lr *= 1e-2
#            elif epoch > 80:
#              epoch_lr *= 1e-1
             #get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = self.train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = self.train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr
                }

                test_feed_dict = {
                    self.test_inptus : self.test_x[0:100],
                    self.test_labels : self.test_y[0:100]
                }

                get_ag = False
                if idx%256  == 0 and idx !=0:
                    get_ag = True
                # update network
                summary_str =  train_loss =  train_accuracy = None
                # if get_ag is False:
                start_time = time.time()
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                end_time = time.time()
                print("Global time: "+ str(end_time-start_time), file=sys.stderr)
                exit(0)
                # else:
                    # _, summary_str, train_loss, train_accuracy = self.sess.run(
                    #     [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                    # self.writer.add_summary(summary_str, counter)

                    # ret = self.sess.run(
                    #     self.ag_list, feed_dict=train_feed_dict)
                    # for x in ret:
                    #     y = np.asarray(x).flatten()
                    #     np.savetxt('grad.csv', y, delimiter=',')    
                    # pass
                # test
                summary_str, test_loss, test_accuracy = self.sess.run(
                    [self.test_summary, self.test_loss, self.test_accuracy], feed_dict=test_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, test_accuracy: %.2f, learning_rate : %.10f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, test_accuracy, epoch_lr))
                
                break
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # save output prediction for each epoch
            self.save_epoch()
            break

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr,self.acc)
#return "{}{}_{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr,"res18-300-am32-cifar10")

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep = self.epoch)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        num_examples  = len( self.test_x)
        total_accuracy = 0
        for offset in range(0,num_examples,100):
            test_feed_dict = {
                self.test_inptus: self.test_x[offset:offset+100],
                self.test_labels: self.test_y[offset:offset+100]
            }
            test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
            # print("test_accuracy: {}".format(test_accuracy))
            total_accuracy += (test_accuracy*100)
        total_accuracy = total_accuracy/num_examples
        print("test_accuracy: {}".format(total_accuracy))

    def save_epoch(self):
        num_examples  = len( self.test_x)
        total_accuracy = 0

        f = open(str(self.model_dir)+"_pred.csv", "ab")
        f1 = open(str(self.model_dir)+"_act.csv", "ab")
        for offset in range(0,num_examples,100):
            test_feed_dict = {
                self.test_inptus: self.test_x[offset:offset+100],
                self.test_labels: self.test_y[offset:offset+100]
            }
            test_accuracy, predictied,actual = self.sess.run([self.test_accuracy,self.predictied,self.actual], feed_dict=test_feed_dict)
            np.savetxt(f,np.asarray(predictied))
            np.savetxt(f1,np.asarray(actual))
            total_accuracy += (test_accuracy*100)
        total_accuracy = total_accuracy/num_examples
        print("test_accuracy: {}".format(total_accuracy))


        f.close()
        f1.close()


# def evaluate(X_data, y_data,summ=None,steps=None):
#     num_examples = len(X_data)
#     total_accuracy = 0
#     total_loss = 0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, BATCH_SIZE):
#     #for offset in range(1):
#         batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#         accuracy,loss = sess.run([accuracy_operation,loss_operation], feed_dict={x: batch_x, y: batch_y})
#         total_accuracy += (accuracy * len(batch_x))
#         total_loss += (loss * len(batch_x))
#     if summ != None:
#         summary_t = sess.run(test_summary,feed_dict={test_loss_ph: total_loss/num_examples, test_acc_ph: total_accuracy/num_examples})
#         summ.add_summary(summary_t,steps)
