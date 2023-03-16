import os
import sys
import numpy as np
from utils import load_input_dude
from Network import Network
import tensorflow as tf
import GCN_model

np.set_printoptions(precision=3)    #控制输出的小数点后个数是3

epoch_size = 80

#prop = 'abl1'
prop = sys.argv[1]

#GPU设置 - 使用下行代码即选择不使用GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


dim1 = 32
dim2 = 256
max_atoms = 75
num_layer = 3
batch_size = 256

learning_rate = 0.001
regularization_scale = 1e-4
beta1 = 0.9
beta2 = 0.98

model_name = 'DUD-E_'+prop

smi_total, prop_total = load_input_dude(prop)
num_total = len(smi_total)
num_test = int(num_total*0.2)
num_train = num_total-num_test
num_eval = int(num_train*0.1)
num_train -= num_eval


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task_type', 'classification', '')
flags.DEFINE_integer('hidden_dim', dim1, '')
flags.DEFINE_integer('latent_dim', dim2, '')
flags.DEFINE_integer('max_atoms', max_atoms, '')
flags.DEFINE_integer('num_layers', num_layer, '')
flags.DEFINE_integer('num_attn', 4, '')
flags.DEFINE_integer('batch_size', batch_size, '')
flags.DEFINE_integer('epoch_size', epoch_size, '')
flags.DEFINE_integer('num_train', num_train, '')
flags.DEFINE_float('regularization_scale', regularization_scale, '')
flags.DEFINE_float('beta1', beta1, '')
flags.DEFINE_float('beta2', beta2, '')
flags.DEFINE_string('optimizer', 'Adam', 'Options : Adam, SGD, RMSProp')
flags.DEFINE_float('init_lr', learning_rate, 'Batch size')


print("Do Single-Task Learning")
print("Hidden dimension of graph convolution layers:", dim1)
print("Hidden dimension of readout & MLP layers:", dim2)
print("Maximum number of allowed atoms:", max_atoms)
print("Batch sise:", batch_size, "Epoch size:", epoch_size)
print("Initial learning rate:", learning_rate, "\t Beta1:", beta1, "\t Beta2:", beta2, "for the Adam optimizer used in this training")

model = Network(FLAGS)
GCN_model.training(model, FLAGS, model_name, smi_total, prop_total)
