import numpy as np
import tensorflow as tf
from ConcreteDropout import ConcreteDropout

def conv1d_with_concrete_dropout(x, out_dim, wd, dd):
    # Conv1D() -> 一维卷积
    output = ConcreteDropout(tf.keras.layers.Conv1D(filters=out_dim,    #filters: 过滤器个数
                                                    kernel_size=1,      #kernel_size : 卷积核的大小
                                                    use_bias=True,
                                                    activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    bias_initializer=tf.contrib.layers.xavier_initializer()),
                             weight_regularizer=wd,
                             dropout_regularizer=dd,
                             trainable=True)(x, training=True)
    return output

def dense_with_concrete_dropout(x, out_dim, wd, dd):
    output = ConcreteDropout(tf.keras.layers.Dense(units=out_dim,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.contrib.layers.xavier_initializer()),
                             weight_regularizer=wd,
                             dropout_regularizer=dd,
                             trainable=True)(x, training=True)
    return output

def attn_matrix(A, X, attn_init):
    '''
    输出是 激活函数tanh( 邻接矩阵A 对应•乘  X * ATTN * X   )

    '''
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F' 
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])
    # einsum -> 爱因斯坦表达式加和
    _X1 = tf.einsum('ij,ajk->aik', attn_init, tf.transpose(X, [0,2,1]))
    similarity = tf.matmul(X, _X1)         # 表面是矩阵乘法, 其实是用两个向量做内积,得到一个方阵, 就是对应元素的距离
    _A = tf.multiply(A, similarity)        # tf.multiply() -> 矩阵对应元素相乘 (对应乘->排除非邻居节点)
    attn_weight = tf.nn.tanh(_A)           # 双曲正切曲线 用作激活函数
    return attn_weight

def get_gate_coeff(X1, X2, dim, label):
    '''
    X1 与 x2分别做tf.layers.dense密集连接层:该层实现了操作：outputs = activation(inputs * kernel + bias)
    之后做X1 + X2 + _b(xavier初始化)
    最后sigmoid之后,返回
    '''
    num_atoms = int(X1.get_shape()[1])
    _b = tf.get_variable('mem_coef-'+str(label), initializer=tf.contrib.layers.xavier_initializer(), shape=[dim], dtype=tf.float32)
    _b = tf.reshape(tf.tile(_b, [num_atoms]), [num_atoms, dim])         # tf.tile(input,multiples)参数:输入,同一维度复制次数 功能:扩张张量,在维度上复制

    X1 = tf.layers.dense(X1, units=dim, use_bias=False)
    X2 = tf.layers.dense(X2, units=dim, use_bias=False)

    output = tf.nn.sigmoid(X1+X2+_b)
    return output

#每一个卷积层//label 是当前的层数
def graph_attn_gate(A, X, attn, out_dim, label, length, num_train):

    #使用AVG合并多头Attention
    X_total = np.zeros( (X.get_shape()[1],out_dim) )

    wd = length**2/num_train
    dd = 2./num_train
    for i in range( len(attn) ):
        _h = conv1d_with_concrete_dropout(X, out_dim, wd, dd)   # 先将特征转化为期望输出的特征维度
        attn_weight = attn_matrix(A, _h, attn[i])    # 计算得到（?,75,75）的一个权重矩阵，视为各个节点之间的重要程度
        _h = tf.nn.relu(tf.matmul(attn_weight, _h))  # 将权重与特征相乘
        X_total = _h + X_total

    _X = tf.nn.relu(X_total/len(attn))      # 使用平均方法，得到最终Attention处理后的结果

    #将X转为与_X输出维度，便于计算最终输出
    dim = int(_X.get_shape()[2])
    if( int(X.get_shape()[2]) != dim ):
        X = tf.layers.dense(X, units=dim, use_bias=False)     # 全连接层 相当于加一个层 输出为dim大小(units:神经元个数)

    #计算权重系数->门机制，保留上一层的特征 减少过平滑
    #通过学习得到权重系数coefficient，视为新一层在最终结果的占比
    coeff = get_gate_coeff(_X, X, dim, label)
    output = tf.multiply(_X, coeff) + tf.multiply(X,1.0-coeff)

    return output

def getAttnInitial(num_attn,i,dim):
    attn_initial = []
    for j in range(num_attn):
        attn_initial.append(tf.get_variable('eaw' + str(i) + '_' + str(j),  # 第 lable 个卷积层的第 j 个自注意力头
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           # xavier_initializer() ->这个初始化器是用来保持每一层的梯度大小都差不多相同
                                           shape=[dim, dim],
                                           dtype=tf.float32))
    return attn_initial

def encoder_gat_gate(X, A, num_layers, out_dim, num_attn, length, num_train):
    # X : Atomic Feature, A : Adjacency Matrix
    _X = X
    for i in range(num_layers):

        #功能:创建num_attn个 outdim*outdim大小的tensor矩阵
        attn_initial_list = getAttnInitial(num_attn,i,out_dim)

        #每一层卷积； out_dim 为输出
        _X = graph_attn_gate(A, _X, attn_initial_list, out_dim, i, length, num_train)

    #END_FOR_i
    return _X

def readout_and_mlp(X, latent_dim, length, num_train):
    # X : [#Batch, #Atom, #Feature] --> Z : [#Batch, #Atom, #Latent] -- reduce_sum --> [#Batch, #Latent]
    # Graph Embedding in order to satisfy invariance under permutation
    wd = length**2/num_train
    dd = 2./num_train

    '''
    加了一个隐层, X(?,75,32) -> Z(?,75,265)
    '''
    Z = tf.nn.relu(conv1d_with_concrete_dropout(X, latent_dim, wd, dd))

    Z = tf.nn.sigmoid(tf.reduce_sum(Z, 1))  #reduce表示降维_sum表示如何降维_1表示沿着哪个方向sum (这里即Z原来 ?,75,256 -> ?,256 75个一起相加)

    # Predict the molecular property
    _Y = tf.nn.relu(dense_with_concrete_dropout(Z, latent_dim, wd, dd))    #Z是(?,256) 输出_Y也是(?,256) 加了一层并且用了一个激活函数

    '''
    Y_mean:为最终图学习后输出的一个值,Sigmoid函数处理后可视为预测正例概率
    Dense() ->全连接层
    tf.keras.layers.Dense(
        inputs=64,  # 输入该网络层的数据
        units=10,  # 输出的维度大小
        activation=None,  # 选择使用的（激活函数）
        use_bias=True,  # 是否使用（偏置项）
        kernel_initializer=None,  # 卷积核的初始化器
        bias_initializer=tf.zeros_initializer(),  # 偏置项的初始化器
        kernel_regularizer=None,  # 卷积核的正则化
        activaty_regularizer=None,  # 偏置项的正则化
        kernel_constraint=None,  # 对主权重矩阵进行约束
        bias_constraint=None,  # 对偏置向量进行约束
        trainable=True,  # 可以设置为不可训练，（冻结）网络层
        name=None,  # 层的名字
        reuse=None  # 是否重复使用参数
        )

    '''
    Y_mean = tf.keras.layers.Dense(units=1,         #最后全连接,输出维度为1
                                   use_bias=True,
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    Y_logvar = tf.keras.layers.Dense(units=1,
                                     use_bias=True,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    return Z, Y_mean, Y_logvar
