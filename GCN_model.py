import numpy as np
import os
import time
from utils import shuffle_two_list, convert_to_graph, split_train_eval_test

import xlwt

from sklearn.metrics import accuracy_score, roc_auc_score

np.set_printoptions(precision=3)    #控制输出的小数点后个数是3


def np_sigmoid(x):
    '''Sigmoid激活函数'''
    return 1. / (1. + np.exp(-x))


def training(model, FLAGS, model_name, smi_total, prop_total):
    print("Start Training")
    num_epochs = FLAGS.epoch_size
    batch_size = FLAGS.batch_size
    init_lr = FLAGS.init_lr  # 学习率
    total_st = time.time()  # 开始时间

    '''划分训练集、验证集、测试集'''
    smi_train, smi_eval, smi_test = split_train_eval_test(smi_total, 0.8, 0.2, 0.1) # 0.1指占总训练集的0.1 train:test = 8:2
    prop_train, prop_eval, prop_test = split_train_eval_test(prop_total, 0.8, 0.2, 0.1)
    prop_eval = np.asarray(prop_eval)
    prop_test = np.asarray(prop_test)
    num_train = len(smi_train)
    num_eval = len(smi_eval)
    num_test = len(smi_test)
    smi_train = smi_train[:num_train]
    prop_train = prop_train[:num_train]
    num_batches_train = (num_train // batch_size) + 1
    num_batches_eval = (num_eval // batch_size) + 1
    num_batches_test = (num_test // batch_size) + 1
    print("Number of-  training data:", num_train, "\t evaluation data:", num_eval, "\t test data:", num_test)

    # 记录信息，保存excel
    epoch_all = []

    trainLoss_all = []
    trainAcc_all = []
    trainAuroc_all = []

    validationLoss_all = []
    validationAcc_all = []
    validationAuroc_all = []

    every_epoch_time = []

    num_sampling = 20
    total_iter = 0  # 总迭代批次

    # num_epoch 指训练几回,比如有一万数据,不是一次性就学完了的,是有一个学习率,然后反复迭代的学这一万个数据,直至最后各个参数接近最优解.(此处固定训练次数100)
    for epoch in range(num_epochs):
        st = time.time()
        lr = init_lr * 0.5 ** (epoch // 10)
        model.assign_lr(lr)
        smi_train, prop_train = shuffle_two_list(smi_train, prop_train)
        prop_train = np.asarray(prop_train)

        # TRAIN
        num = 0
        train_loss = 0.0
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])

        # 每一轮训练时,不是把所有数据都读进去,一次处理一批(batch)
        for i in range(num_batches_train):
            num += 1
            st_i = time.time()
            total_iter += 1
            A_batch, X_batch = convert_to_graph(smi_train[i * batch_size:(i + 1) * batch_size], FLAGS.max_atoms)
            Y_batch = prop_train[i * batch_size:(i + 1) * batch_size]

            # Y_mean就是sigmoid之前的Y_predict(正例预测概率)
            Y_mean, _, loss = model.train(A_batch, X_batch, Y_batch)
            train_loss += loss
            Y_pred = np_sigmoid(Y_mean.flatten())
            # np.concatenate() 数组拼接
            Y_pred_total = np.concatenate((Y_pred_total, Y_pred), axis=0)
            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)

            et_i = time.time()
            # print('第',(i+1),'/',num_batches_train,'次训练结束 本次共耗时:',(et_i-st_i) ,'秒')

        train_loss /= num
        # np.around() 四舍五入, Y_pred_total就是预测出来为正例的概率,四舍五入一下就是正负例的结果
        train_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
        # AUC ：曲线下面积（Area Under the Curve）
        # AUROC ：接受者操作特征曲线下面积（Area Under the Receiver Operating Characteristic curve）
        train_auroc = 0.0
        try:
            # roc_auc_score() -> Sklearn里面的函数 返回AUC值
            train_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
        except:
            train_auroc = 0.0

        # Eval
        Y_pred_total = np.array([])
        Y_batch_total = np.array([])
        num = 0
        eval_loss = 0.0
        for i in range(num_batches_eval):
            A_batch, X_batch = convert_to_graph(smi_eval[i * batch_size:(i + 1) * batch_size], FLAGS.max_atoms)
            Y_batch = prop_eval[i * batch_size:(i + 1) * batch_size]

            # 蒙特卡洛采样
            P_mean = []
            for n in range(3):
                num += 1
                Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
                eval_loss += loss
                P_mean.append(Y_mean.flatten())

            P_mean = np_sigmoid(np.asarray(P_mean))
            mean = np.mean(P_mean, axis=0)

            Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
            Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)

        eval_loss /= num
        eval_accuracy = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
        eval_auroc = 0.0
        try:
            eval_auroc = roc_auc_score(Y_batch_total, Y_pred_total)
        except:
            eval_auroc = 0.0

        et = time.time()

        #打印结果
        print('[',model_name,"]\t  -",epoch,"/",num_epochs,"     \tTime:",round( et - st,2))
        print("Loss        \tTrain:", round(train_loss, 3), "\tEvaluation:", round(eval_loss, 3))
        print("Accuracy    \tTrain:", round(train_accuracy, 3), "\tEvaluation:", round(eval_accuracy, 3))
        print("AUROC       \tTrain:", round(train_auroc, 3), "\tEvaluation:", round(eval_auroc, 3))

        # 保存数据
        epoch_all.append(epoch)
        trainLoss_all.append(round(train_loss, 3))
        trainAcc_all.append(round(train_accuracy, 3))
        trainAuroc_all.append(round(train_auroc, 3))
        validationLoss_all.append(round(eval_loss, 3))
        validationAcc_all.append(round(eval_accuracy, 3))
        validationAuroc_all.append(round(eval_auroc, 3))
        every_epoch_time.append(round(et - st, 2))

    total_et = time.time()
    print("Finish training! Total required time for training : ", (total_et - total_st))

    # Test
    test_st = time.time()
    Y_pred_total = np.array([])
    Y_batch_total = np.array([])
    ale_unc_total = np.array([])
    epi_unc_total = np.array([])
    tot_unc_total = np.array([])
    num = 0
    test_loss = 0.0
    for i in range(num_batches_test):
        num += 1
        A_batch, X_batch = convert_to_graph(smi_test[i * batch_size:(i + 1) * batch_size], FLAGS.max_atoms)
        Y_batch = prop_test[i * batch_size:(i + 1) * batch_size]

        #蒙特卡洛擦痒 设置T为20次
        P_mean = []
        for n in range(num_sampling):
            Y_mean, _, loss = model.test(A_batch, X_batch, Y_batch)
            P_mean.append(Y_mean.flatten())

        P_mean = np_sigmoid(np.asarray(P_mean))

        mean = np.mean(P_mean, axis=0)
        ale_unc = np.mean(P_mean * (1.0 - P_mean), axis=0)
        epi_unc = np.mean(P_mean ** 2, axis=0) - np.mean(P_mean, axis=0) ** 2
        tot_unc = ale_unc + epi_unc

        Y_batch_total = np.concatenate((Y_batch_total, Y_batch), axis=0)
        Y_pred_total = np.concatenate((Y_pred_total, mean), axis=0)
        ale_unc_total = np.concatenate((ale_unc_total, ale_unc), axis=0)
        epi_unc_total = np.concatenate((epi_unc_total, epi_unc), axis=0)
        tot_unc_total = np.concatenate((tot_unc_total, tot_unc), axis=0)

        test_acc = accuracy_score(Y_batch_total, np.around(Y_pred_total).astype(int))
        test_auroc = roc_auc_score(Y_batch_total, Y_pred_total)


    test_et = time.time()
    print("Finish Testing, Total time for test:", (test_et - test_st))

    # 训练-测试完成 保存Excel文件：
    workbook = xlwt.Workbook(encoding='utf-8')
    trainSheet = workbook.add_sheet('Train')
    trainSheet.write(0, 0, 'epoch')
    trainSheet.write(0, 1, 'trainLoss')
    trainSheet.write(0, 2, 'trainAcc')
    trainSheet.write(0, 3, 'trainAuroc')
    trainSheet.write(0, 4, 'validationLoss')
    trainSheet.write(0, 5, 'validationAcc')
    trainSheet.write(0, 6, 'validationAuroc')
    trainSheet.write(0, 7, 'train-time')
    for i in range(num_epochs):
        trainSheet.write(i + 1, 0, epoch_all[i])
        trainSheet.write(i + 1, 1, trainLoss_all[i])
        trainSheet.write(i + 1, 2, trainAcc_all[i])
        trainSheet.write(i + 1, 3, trainAuroc_all[i])
        trainSheet.write(i + 1, 4, validationLoss_all[i])
        trainSheet.write(i + 1, 5, validationAcc_all[i])
        trainSheet.write(i + 1, 6, validationAuroc_all[i])
        trainSheet.write(i + 1, 7, every_epoch_time[i])

    testSheet = workbook.add_sheet('Test')
    testSheet.write(0, 0, '')
    testSheet.write(0, 1, 'test-acc')
    testSheet.write(0, 2, 'test-auroc')
    testSheet.write(0, 3, 'test-time')
    testSheet.write(1, 0, 'ans:')
    testSheet.write(1, 1, round(test_acc, 3))
    testSheet.write(1, 2, round(test_auroc, 3))
    testSheet.write(1, 3, round((test_et - test_st), 2))

    # 保存
    filepath = './result/'
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    workbook.save(filepath + model_name + '_result.xls')

    return