import numpy as np
from utils import load_input_tox21
from Network import Network
import tensorflow as tf
import os
import sys
import GCN_model

np.set_printoptions(precision=3)    #控制输出的小数点后个数是3

#训练回合数
epoch_size = 100

#选择子数据集

#prop = 'nr-ahr'
prop = sys.argv[1]

#GPU设置
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#设置超参数
dim1 = 32
dim2 = 256
max_atoms = 75
num_layer = 4
batch_size = 256

learning_rate = 0.001
regularization_scale = 1e-4
beta1 = 0.9
beta2 = 0.98

model_name = 'tox21_'+prop

#导入训练集
smi_total, prop_total = load_input_tox21(prop, max_atoms)
num_total = len(smi_total)
#计算num_train数量
num_test = int(num_total*0.2)
num_train = num_total-num_test
num_eval = int(num_train*0.1)
num_train -= num_eval

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task_type', 'classification', '任务类型')
flags.DEFINE_integer('hidden_dim', dim1, '卷积层处理后最终输出的维度')
flags.DEFINE_integer('latent_dim', dim2, '读出层中隐藏层的大小')
flags.DEFINE_integer('max_atoms', max_atoms, '可接受最大原子数')
flags.DEFINE_integer('num_layers', num_layer, '卷积层数')
flags.DEFINE_integer('num_attn', 4, '多头注意力机制头数')
flags.DEFINE_integer('batch_size', batch_size, '训练批次大小')
flags.DEFINE_integer('epoch_size', epoch_size, '训练回合数')
flags.DEFINE_integer('num_train', num_train, '训练集数量')
flags.DEFINE_float('regularization_scale', regularization_scale, '正则缩放')
flags.DEFINE_float('beta1', beta1, 'Adam优化器参数1')
flags.DEFINE_float('beta2', beta2, 'Adam优化器参数2')
flags.DEFINE_string('optimizer', 'Adam', '优化器')
flags.DEFINE_float('init_lr', learning_rate, '学习率')


print("Do Single-Task Learning")
print("Hidden dimension of graph convolution layers:", dim1)
print("Hidden dimension of readout & MLP layers:", dim2)
print("Maximum number of allowed atoms:", max_atoms)
print("Batch sise:", batch_size, "Epoch size:", epoch_size)
print("Initial learning rate:", learning_rate, "\t Beta1:", beta1, "\t Beta2:", beta2, "for the Adam optimizer used in this training")

model = Network(FLAGS)
GCN_model.training(model, FLAGS, model_name, smi_total, prop_total)