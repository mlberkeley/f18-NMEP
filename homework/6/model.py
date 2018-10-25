import tensorflow as tf
import yaml
import os
from image_dataset import ImageDataset
from utils import AttrDict
from tensorflow.contrib.slim.nets import resnet_v2
from pprint import pprint
from glob import glob

class TransferModel(object):
    '''Transfer model object for image classification.'''

    # This is the meat of the code and where all the interesting operations happen, 
    # in this TransferModel class. This is also where you'll be filling in the most content!
    
    def __init__(self, sess, flags):
    `   # This function is finished.
        # Pass in the tensorflow session, and the logistical gflags.
        self.sess = sess

        # Read in the config.yaml and get it into a format thats easy to work with

        stream = open(flags.config_yaml, 'r')
        config = yaml.load(stream)
        # config is a dict here. but indexing is annoying to write and read (config['learning_rate'])
        # so AttrDict takes in a dict and spits out an object you can access attributes of: config.learning_rate
        # For those interested in some spicy python, check out utils.py for the definition of AttrDict but has
        # nothing to do with ML. 
        self.config = AttrDict(config)
        # keep the config dict though, makes writing it easier later on ;) 
        self.config_d = config
        
        self.model_id = flags.model_id
        self.data_dir = flags.data_dir
        self.set_model_params(self.config)

        if not flags.train:
            #If we're not training, make a test dataset for inference
            self.test_dataset = ImageDataset(
                data_dir=os.path.join(self.data_dir, 'test'),
                h=self.height,
                w=self.width,
                batch_size=self.batch_size,
                crop_proportion=None
            )
        # Path for the ckpt of pretrained resnet
        self.path = 'pretrained_models/resnet_v2_50.ckpt'
        self.build_graph()

    def set_model_params(self, config):
    `   # This function is finished.
        # The code in here just grabs things from the config and puts them into the model
        # to make them slightly less annoying to work with

        self.num_classes = config.num_classes
        self.height = config.height
        self.width = config.width
        self.write_images = config.write_images
        self.batch_size = config.batch_size
        
    def test_net(self, inputs):
    `   # This function is finished.
        # Make a call to get output of resnet given some inputs    
        net, end_points = resnet_v2.resnet_v2_50(
            inputs,
            num_classes=self.num_classes,
            is_training=False,
            global_pool=True,
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope="resnet_v2_50"
        )
        return net, end_points

    def train_net(self, inputs, is_training=True):
        # This function is finished.
        
        net, end_points = resnet_v2.resnet_v2_50(
            inputs,
            num_classes=self.num_classes,
            is_training=True,
            global_pool=True,
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope="resnet_v2_50"
        )
        #what are endpoints? try pprint.pprint(endpoints) ;) 
        return net, end_points
            
            
    def build_graph(self):
        # Make some placeholders first: what do you need to pass into the graph in order to get out what you want?
        # Hints: there are three necessary for training 
         
        #TODO: write down your output given placeholders
        #Disclaimer: not necessarily one line:
        y_hat = ...

        # Why do we need this? For inference.
        #TODO: give this variable a better name

        self.sOmEtHinG = tf.nn.softmax(y_hat)

        
        self.loss = tf.reduce_mean( #TODO: Make a call to softmax cross entropy. Sorry, this is basically reading documentation. But you'll be using that function a lot, and you should know the particulars of how you must call it
        )
        self.accuracy = #Try implementing this yourself without looking it up, good practice being familiar with tf 

        loss_sum = tf.summary.scalar('loss', self.loss)
        acc_sum = ... 

        if self.write_images:
            # something to consider: Why are we being particular about whether we record images or not? <hmm>
            #TODO: write down the name of your images/what you want to visualize in tboard
            img_sum = tf.summary.image('train_images', ?)

        self.summary = tf.summary.merge_all()
        
        # Perhaps it would make more sense to put these below, in make_train_graph. try it out
        self.opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=self.config.beta1)
        self.saver = tf.train.Saver(max_to_keep=10)

    def make_train_graph(self):
        
        #TODO: write down which variables we'll train when fine-tuning
        # Using pprint would help here:
        vars_to_train = ... 

        if self.config.weight_decay > 0:
            #TODO: Implement L2 weight decay!  
        
        self.train_step = self.opt.minimize(self.loss, var_list=vars_to_train)

        if self.model_id != -1:
            start_epoch = self.restore()
        else:
            self.init_new_model()
            start_epoch = 0

        log_dir = os.path.join(self.model_dir, 'logs/train')
        val_dir = os.path.join(self.model_dir, 'logs/val')
        
        self.train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        self.val_writer = tf.summary.FileWriter(val_dir, self.sess.graph)

        self.train_dataset = ImageDataset(
            data_dir=os.path.join(self.data_dir, 'train'),
            h=self.height,
            w=self.width,
            batch_size=self.batch_size,
            crop_proportion=self.config.crop_proportion
        )
        #TODO: validation dataset, whats diff from train dataset?
        self.val_dataset = ...

        return start_epoch
        

    def train(self):
        start_epoch = self.make_train_graph()
        idx = 0
        print("Starting training")

        #TODO: Implement training loop:

        for epoch in range(start_epoch, self.config.num_epochs):
            
            #at some point, print these out:
            print("training: idx {} loss {} acc {}".format(idx, ... )
            print("val stats: idx {} loss {} acc {}".format(idx, ... )

            self.save(idx, epoch)
        print("Done Training")

    def predict(self, path):
        #TODO: Get image, run through network
        # What placeholders do you really need?

        feed_dict= ...
        probabilities = self.sess.run(?, feed_dict=feed_dict)
        print("Hartebeest: {}\nDeer: {}\nSheep: {}\n".format(...))

    def init_new_model(self):
        assert self.model_id < 0
        try:
            self.model_id = 1 + max([
                int(f.split('/')[-1][1:]) for f in glob('experiments/m*')
            ])
        except ValueError:
            self.model_id = 0
            # try/except here is pretty hacky

        self.model_dir = os.path.join('./experiments', 'm'+str(self.model_id))
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        with open(os.path.join(self.model_dir, 'config.yaml'), 'w') as out:
            # We want to save which hyperparameters we're using for each model- v important
            yaml.dump(self.config_d, out, default_flow_style=False)

        #TODO: Initialize variables.
        #TODO: Make a list of variables with which to load from saver
        #TODO: Make a Saver, pass in which variables to load
        #TODO: Restore from pretrained ckpt

    def restore(self):
        # This function is finished
        # Restore latest model from some directory:

        assert self.model_id >= 0, "Trying to restore a negative model id"
        self.model_dir = os.path.join('./experiments', 'm{}'.format(self.model_id))

        stream = open(os.path.join(self.model_dir, 'config.yaml'))
        loaded_train_config = yaml.load(stream)

        print("Loaded train config:")
        pprint(loaded_train_config)
        self.sess.run(tf.global_variables_initializer())
        checkpoint_dir = os.path.join(self.model_dir, 'ckpts')
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        start_epoch = 1 + int(checkpoint_path.split('.ckpt')[1].split('-')[1])
        print("Restoring from {} with epoch {}".format(checkpoint_path, start_epoch))
        self.saver.restore(self.sess, checkpoint_path)
        return start_epoch

    def save(self, idx, epoch):
        # This function is finished
        # Just a wrapper for saving

        self.saver.save(
            self.sess,
            os.path.join(self.model_dir, 'ckpts/model.ckpt'),
            global_step=idx,
            write_meta_graph=not bool(epoch)
        )
