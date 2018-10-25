import os
import tensorflow as tf
from model import TransferModel
import yaml

# These are google's gflags. You can use different things here, like argparse.
# You pass these into your program like arguments to command e.g. python3 main.py --train=False
# Format: FLAGS.DEFINE_<type>(name, default value, description)


FLAGS = tf.app.flags
FLAGS.DEFINE_boolean("train", False, "whether to train or infer")
FLAGS.DEFINE_string("data_dir", "./data/", "directory where data is stored")

# We use gflags for the logistical parts of the code, and config.yaml for all the network-y parameters,
# Like hyperparameters and configurations. You can look at config.yaml, which is provided as a default
# for formatting of .yaml files. The idea for the separation is that you can run the code across a bunch 
# of hyperparameters defined by unique config.yaml's for hyperparameter searching
FLAGS.DEFINE_string("config_yaml", "./config.yaml", "hyperparameters for training")
FLAGS.DEFINE_integer("model_id", -1, "If positive: restores the model given by the id, else new model. required for inference")

FLAGS.DEFINE_string("image_path", None, "If not none: runs prediction model on this image")

FLAGS=FLAGS.FLAGS # probably my favorite line

def main(_):
    with tf.Session() as sess:
        # We'll make the session out here.
        # Make a new model object

        model = TransferModel(sess, FLAGS)

        if FLAGS.train:
            # this is how you access the properties of FLAGS--it's like an object with all the flags as  
            model.train()
        else:
            if FLAGS.image_path is not None:
                model.restore()
                model.predict(FLAGS.image_path)

if __name__ == '__main__':
    # run this shit
    tf.app.run()

