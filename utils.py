import numpy as np
import random
from rdkit import Chem


def shuffle_two_list(list1, list2):
    '''
    功能:同时对两个list做相同的shuffle, 对应位置不变
    '''

    # 使用zip打包两个对象成一个元组,python3中是返回一个对象,需要手动list转换.
    # a = [1,2,3] b = [4,5,6], list(zip(a,b) = [(1, 4), (2, 5), (3, 6)]
    list_total = list(zip(list1,list2))
    random.shuffle(list_total)

    # 加*,zip的逆操作
    list1, list2 = zip(*list_total)
    return list1, list2


def load_input_HIV():
    smi_list = []
    prop_list = []
    # Actives
    f = open('./data/HIV/HIV_all.txt', 'r')
    lines = f.readlines()
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        smi_list.append(smi)
        prop_list.append(prop)
    return smi_list, prop_list


def load_input_dude(target_name):
    smi_list = []
    prop_list = []
    f = open('./data/dude/'+target_name+'_all.txt', 'r')
    lines = f.readlines()
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        smi_list.append(smi)
        prop_list.append(prop)
    return smi_list, prop_list


def load_input_tox21(tox_name, max_atoms):
    '''
    功能: 读入tox21数据集数据,筛选掉无法生成Mol分子的行,仅读入 原子数<=max_atoms 的分子.
    '''
    f = open('./data/tox21/'+tox_name+'_all.txt', 'r')
    lines = f.readlines()
    smi_list = []
    prop_list = []
    for l in lines:
        smi = l.split(',')[0]
        prop = float(l.split(',')[1].strip())
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        else:
            if( m.GetNumAtoms() < max_atoms+1 ):
                smi_list.append(smi)
                prop_list.append(prop)
    return smi_list, prop_list


def split_train_eval_test(input_list, train_ratio, test_ratio, eval_ratio):
    '''
    功能: 按照比例划分训练集、测试集、验证集
    '''

    #先分为训练集和测试集; 之后训练集再分为真正的训练集和验证集
    num_total = len(input_list)
    num_test = int(num_total*test_ratio)
    num_train = num_total-num_test
    num_eval = int(num_train*eval_ratio)
    num_train -= num_eval

    train_list = input_list[:num_train]
    eval_list = input_list[num_train:num_train+num_eval]
    test_list = input_list[num_train+num_eval:]
    return train_list, eval_list, test_list

def convert_to_graph(smiles_list, max_atoms):
    '''
    功能: 将多个SMILES格式的分子转换为邻接矩阵
    输出:Adj_list、feature_list
    '''

    adj = []
    features = []
    for i in smiles_list:
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)         #Adj 根据Mol格式转换邻接矩阵
        # Feature
        if( iAdjTmp.shape[0] <= max_atoms):
            # Feature-preprocessing
            iFeature = np.zeros((max_atoms, 58))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((max_atoms, max_atoms))
            #np.eye 生成一个对角矩阵
            #array相加: 正常的矩阵相加
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), 1))

    features = np.asarray(features)
    adj = np.asarray(adj)
    return adj, features


def adj_k(adj, k):
    '''
    功能: 输出adj的k次幂
    '''
    ret = adj
    for i in range(0, k-1):
        # np.dot: 如果是一维数组, 返回点积; 如果是矩阵 返回矩阵的乘积
        ret = np.dot(ret, adj)
    return convert_adj(ret)


def convert_adj(adj):
    '''
    功能: 通过两步操作,一、与0异或 二、用1减 的两次取反操作
    将一个矩阵中所有非零元素置为1: 0仍为0,大于1的置为1
    '''
    dim = len(adj)
    # flatten-> 把一个矩阵或者数组降到一维,默认按横着方向排列
    a = adj.flatten()
    #zeros(dim*dim) 输出一个 (dim^2,)的一维向量
    b = np.zeros(dim*dim)
    # np.equal 输入两个矩阵或向量,对应位置相等的为True,其余为False
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))

    return d


def atom_feature(atom):
    '''
    输入atom对象,输出这个原子的特征编码

    Args:
        atom: SMIELS内置atom对象

    Returns:
        长度为58的array作为特征编码,one-hot格式(原子种类、成键数目、相连氢原子数、隐式化合价、是否在芳香烃内)
    '''
    return np.array(
                    #原子种类的one-hot编码
                    one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    #原子的度，这里的度指的是这个原子参与形成键的数目
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    #与该原子相连的氢原子数量
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    #获取原子隐式化合价
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    #该原子是否在芳香烃内
                    [atom.GetIsAromatic()]
                    )    # (40, 6, 5, 6, 1)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''
    返回一个原子的one-hot编码,例:元素符号相应位为1,其余为0
    '''
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
