# 代码运行文档》



## 1 运行环境:



Python：3.6.13、conda:4.10.1

| 主要依赖包名   | 版本号      |
| -------------- | ----------- |
| tensorflow     | 1.9.0       |
| tensorflow-gpu | 1.9.0       |
| numpy          | 1.19.2      |
| scikit-learn   | 0.24.1      |
| rdkit          | 2020.09.1.0 |





## 2 文件结构说明

![image-20210607194515869](https://mypicture-bucket.oss-cn-beijing.aliyuncs.com/img-PicGo/image-20210607194515869.png)

**文件夹:**

data: 存放数据集

result: 运行结果的excel表示, 程序运行后自动生成



**文件:**

blocks.py: 定义单独的一维卷积层、 注意力层、 门机制参数计算方式等等

ConcreteDropout.py: 定义可自调节的Dropout概率

GCN_model.py : 最顶层的模型文件, train_xxx.py 定义好相关超参数后, 直接GCN_model定义的函数.

Network.py: 定义GCN的网络结构

**train_dede.py**: 定义DUD-e数据集训练的相关超参数, 运行即训练DUD-e数据集

**train_tox21.py:** 定义Tox21数据集训练的相关超参数, 运行即训练Tox21数据集

**train_HIV.py:** 定义HIV数据集训练的相关超参数, 运行即训练HIV数据集

utils.py : 定义节点特征向量编码, 定义数据集预处理操作等.







## 3 直接测试:

本次毕设全部代码已经部署至HILab-GPU服务器中,且环境已经部署完毕,  按以下步骤, 即可直接进行测试:



### 3.1 运行步骤(基于HILab服务器)

1. 连接吉林大学校园网, 通过 ssh 登录服务器() 

   > 账号：
   > 密码：

   

2.  进入 **GCN_Code**文件夹

```shell
cd ~/GCN_Code
```



3. 使用conda 进入 **GCN_tensorflow_py36** 环境

```shell
conda activate GCN_tensorflow_py36
```



4. 运行代码(下为三个数据集)

```python
python3.6 train_tox21.py nr-ahr

python3.6 train_dude.py egfr

python3.6 train_HIV.py
```

> 运行成功后, 结果会自动保存至 result文件夹下



### 3.2 注意:

1  除HIV数据集外, tox21与DUD-e均包含子数据集, 运行时需通过参数形式指定需运行的数据集, 例:

```python
python3.6 train_tox21.py nr-ahr
```

> 文件夹中也提供了shell脚本, 可一次运行所有子数据集:
>
> 运行指令:       ./task-GCN-tox21
>
> ![image-20210607191050998](https://mypicture-bucket.oss-cn-beijing.aliyuncs.com/img-PicGo/image-20210607191050998.png)



2 实验室服务器GPU资源有限, 当显存不足时无法运行模型

使用:

```shell
nvidia-smi
```

查看当前GPU使用情况

如果GPU被别人迟迟占着不放, 或未在服务器进行测试, 测试机无GPU, 可以打开python文件, 添加代码:

![image-20210607200325444](https://mypicture-bucket.oss-cn-beijing.aliyuncs.com/img-PicGo/image-20210607200325444.png)

添加此行代码后, 即使用CPU进行计算,



### 4 联系方式

如有任何问题, 请联系:
