import tensorflow as tf
import numpy as np
import itertools
import time
from paint_figure import Visualization
import shujuchuli
import os
import shujuchuli_MFPT
import random
import shujuchuli_JNU
from sklearn.metrics import *


'''
DHA代码来自：
https://github.com/crazy3water/Dynamic-Hypersphere-Algorithm-for-Classification
'''

class DHA_SDA():
    def __init__(self, unit1, unit2, unit3, model_name='sda_scope', train_step=10, fault_num=10,
                 batch_size=500, lr=0.001, R_constant = 1.0, P=1.0, P_R2Cm=1.0, P_class=0.0001):
        """
        :param DataFram:  数据集
        :param train_step:训练步数
        :param R_constant:初始化半径
        :param lr:        优化器学习率
        :param P:         每个球体的惩罚系数
        :param P_R2Cm:    球体之间的惩罚系数
        :param P_class:   加速收敛系数
        """
        self.train_step = train_step
        self.lr = lr
        self.P = P
        self.P_R2Cm = P_R2Cm
        self.P_class = P_class
        self.R_constant = R_constant

        # 编码器部分
        self.lambda1 = 1e-4  # 权重衰减项
        self.beta = 0.002  # 稀疏惩罚项系数
        self.rho = 0.2  # 稀疏因子
        self.keep_prop = 0.98
        self.batch_size = batch_size
        self.unit1 = unit1
        self.unit2 = unit2
        self.unit3 = unit3
        self.pop = np.array([600, self.unit1, self.unit2, self.unit3, 1])
        self.model_name = model_name
        self.dha_sda = tf.Graph()
        self.fault_num = fault_num

        self.data_space = {}
        self.index_space = {}
        self.activations = []
        self.names = {}
        self.split_num = []  # 每个类别的数据的个数

        # lrp
        self.weights = []
        self.activations = []


    '''
    加入稀疏性约束，rho_hat是输入层所有神经元在隐层第j个神经元上激活值的平均，rho是稀疏因子
    稀疏自编码器代码： 
    https://github.com/summersunshine1/datamining/blob/master/sparseencoder/sparseAutoEncoder.py
    '''
    # 编码器的损失函数
    def sda_loss(self, data, outputdata, hidden, hidden_w, output_w):

        error_loss = tf.reduce_mean(tf.square(outputdata - data))
        tf.add_to_collection("losses", error_loss)
        # 参数L2正则化：在原来的损失函数基础上加上权重参数的平方和： 限制参数过多或者过大，避免模型更复杂，可以降低过拟合
        regularizer = tf.contrib.layers.l2_regularizer(self.lambda1)
        regularization = regularizer(hidden_w) + regularizer(output_w)
        tf.add_to_collection("losses", regularization)
        # 加入稀疏性约束
        rho_hat = tf.reduce_mean(hidden, 0)
        Sparse_cost = self.beta * (tf.reduce_sum(self.rho * tf.log(self.rho / rho_hat) + (1 - self.rho) *
                                                 tf.log((1 - self.rho) / (1 - rho_hat))))
        tf.add_to_collection("losses", Sparse_cost)

        loss_sae = tf.add_n(tf.get_collection("losses"))
        optimizer_sae = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_sae)
        return loss_sae, optimizer_sae


    # 在圆心外惩罚
    def g1n_term(self, var, center, Rm):
        g1n = tf.linalg.norm(var - center, axis=1) - Rm  # 点到球心的距离 - 半径 = 点到球的距离
        g1n_max = tf.clip_by_value(g1n, 0, 1e10)
        penalty = tf.reduce_mean(g1n_max)  # if res>0, penalty = res else penalty = 0
        return penalty

    # 在圆心内惩罚
    def g2n_term(self, var, center, Rm):
        g2n = Rm - tf.linalg.norm(var - center, axis=1)  # 半径 - 点到球心的距离 = 点到球边缘的距离
        g2n_max = tf.clip_by_value(g2n, 0, 1e10)
        penalty = tf.reduce_mean(g2n_max)  # if res>0,penalty = res else penalty = 0
        return penalty


    # DHA分类器的损失函数
    def dha_loss(self):
        with self.dha_sda.as_default():
            with tf.name_scope(self.model_name):
                with tf.name_scope('loss_pow'):
                    split_list = np.arange(0, len(self.split_num))
                    for i in range(len(self.split_num)):
                        with tf.name_scope('loss_pow{}'.format(i)):
                            split_list_ = np.delete(split_list, i)
                            if len(split_list_) == 1:
                                U_ = self.names['U{}'.format(split_list_[0])]
                            else:
                                U_ = tf.concat([self.names['U{}'.format(j)] for j in split_list_], 0)

                            g1n1 = self.g1n_term(self.names['U{}'.format(i)],  # layer：weight * v + bias
                                                 self.names['Cm{}'.format(i)],  # 球心
                                                 self.names['R{}'.format(i)])  # 半径
                            g2n1 = self.g2n_term(U_,
                                                 self.names['Cm{}'.format(i)],
                                                 self.names['R{}'.format(i)])

                            Rn = tf.where(tf.greater(np.float64(0), self.names['R{}'.format(i)]),
                                          self.names['R{}'.format(i)], 0)
                            self.names['loss{}_pow'.format(i)] = tf.pow(g1n1, 2) + tf.pow(g2n1, 2) + tf.pow(Rn, 2)

                    loss_pow = 0
                    for i in range(len(self.split_num)):
                        loss_pow = loss_pow + self.names['loss{}_pow'.format(i)]


                with tf.name_scope('lossR2Cm'):
                    combine = list(itertools.combinations(np.arange(0, len(self.split_num)).tolist(), 2))
                    loss_R2Cm = 0
                    for i in combine:
                        with tf.name_scope('lossR2Cm{}{}'.format(i[0], i[1])):
                            Cm_normal = (self.names['R{}'.format(i[0])] + self.names['R{}'.format(i[1])]) \
                                        - tf.linalg.norm(self.names['Cm{}'.format(i[0])] - self.names['Cm{}'.format(i[1])])
                            loss_R2Cm = loss_R2Cm + tf.where(tf.greater(Cm_normal, 0), Cm_normal, 0)


                with tf.name_scope('loss_class'):
                    loss_class = 0.0
                    for i in range(len(self.split_num)):
                        loss_class = loss_class + tf.linalg.norm(self.names['U{}'.format(i)] - self.names['Cm{}'.format(i)])

                with tf.name_scope('loss_all'):
                    self.loss_all = self.P * (loss_pow) + self.P_R2Cm * loss_R2Cm + self.P_class * loss_class

            return self.loss_all


    def gen_model(self, train_data, train_label):
        with self.dha_sda.as_default():
            with tf.name_scope(self.model_name):

                for class_i in range(self.fault_num):
                    i_d = np.argwhere(train_label == class_i)
                    i_d = i_d[:, 0]
                    self.split_num.append(len(i_d))
                    self.data_space['data{}'.format(class_i)] = train_data[i_d]
                    self.index_space['index{}'.format(class_i)] = i_d  # 二维


                with tf.name_scope('InitVariable'):
                    for i in range(len(self.split_num)):
                        with tf.name_scope('V{}'.format(i)):
                            self.names['V{}'.format(i)] = tf.placeholder(
                                tf.float64, shape=[None, train_data.shape[1]], name='Input{}'.format(i))


                    with tf.name_scope('weight'):
                        self.hidden_w1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[0], self.pop[1]]),
                                                        trainable=True), tf.float64, name='hidden_w1')  # float32
                        output_w1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[1], self.pop[0]]),
                                                        trainable=True), tf.float64, name='output_w_1')
                        self.hidden_w2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[1], self.pop[2]]),
                                                        trainable=True), tf.float64, name='hidden_w2')  # float32
                        output_w2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[2], self.pop[1]]),
                                                        trainable=True), tf.float64, name='output_w_2')
                        self.hidden_w3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[2], self.pop[3]]),
                                                        trainable=True), tf.float64, name='hidden_w3')  # float32
                        output_w3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[3], self.pop[2]]),
                                                     trainable=True), tf.float64, name='out_w_3')

                        self.weights = [self.hidden_w1, self.hidden_w2, self.hidden_w3]


                    with tf.name_scope('bias'):
                        self.hidden_b1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[1]]),
                                                        trainable=True), tf.float64, name='hidden_b1')
                        output_b1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[0]]),
                                                        trainable=True), tf.float64, name='output_b_1')
                        self.hidden_b2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[2]]),
                                                        trainable=True), tf.float64, name='hidden_b2')
                        output_b2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[1]]),
                                                        trainable=True), tf.float64, name='output_b_2')
                        self.hidden_b3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[3]]),
                                                        trainable=True), tf.float64, name='hidden_b3')
                        output_b3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[2]]),
                                                     trainable=True), tf.float64, name='out_b_3')


                    # R_constant:初始化半径
                    with tf.name_scope('R'):
                        for i in range(len(self.split_num)):
                            self.names['R{}'.format(i)] = tf.Variable(initial_value=self.R_constant,
                                                                 dtype=tf.float64, name='R{}'.format(i), trainable=True)


                with tf.name_scope('encoder_layer'):

                    for i in range(len(self.split_num)):
                        hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(self.names['V{}'.format(i)], self.hidden_w1), self.hidden_b1))
                        output_ae1 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, output_w1), output_b1))
                        hidden1 = tf.nn.dropout(hidden1, keep_prob=self.keep_prop)
                        hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, self.hidden_w2), self.hidden_b2))
                        output_ae2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, output_w2), output_b2))
                        hidden2 = tf.nn.dropout(hidden2, keep_prob=self.keep_prop)
                        hidden3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, self.hidden_w3), self.hidden_b3))
                        out_ae3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden3, output_w3), output_b3))
                        self.names['U{}'.format(i)] = tf.nn.dropout(hidden3, keep_prob=self.keep_prop)


                        # 逐层训练
                        self.loss_ae1, self.optimizer_ae1 = self.sda_loss(
                            self.names['V{}'.format(i)], output_ae1, hidden1, self.hidden_w1, output_w1)

                        self.loss_ae2, self.optimizer_ae2 = self.sda_loss(
                                hidden1, output_ae2, hidden2, self.hidden_w2, output_w2)

                        self.loss_ae3, self.optimizer_ae3 = self.sda_loss(
                            hidden2, out_ae3, self.names['U{}'.format(i)], self.hidden_w3, output_w3)

                    self.activations = [self.names['V{}'.format(i)], hidden1, hidden2, self.names['U{}'.format(i)]]


                with tf.name_scope('circle'):
                    for i in range(len(self.split_num)):
                        self.names['Cm{}'.format(i)] = tf.reduce_mean(self.names['U{}'.format(i)], axis=0)



    def fit(self, train_data, train_label, tree_i):
        with self.dha_sda.as_default():
            with tf.name_scope(self.model_name):

                self.gen_model(train_data, train_label)
                cl_loss = self.dha_loss()

                with tf.name_scope('Optimizer'):
                    learning_rate = tf.train.exponential_decay(self.lr, self.train_step, decay_steps=100,
                                                               decay_rate=0.8)
                    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cl_loss)

                for weight in self.weights:
                    tf.add_to_collection('LRP_SDA_weights'+str(tree_i), weight)

                for act in self.activations:
                    tf.add_to_collection('LRP_SDA_activations'+str(tree_i), act)
                # we = tf.get_collection('LRP_SDA_weights')
                # print(we)
                # ac = tf.get_collection('LRP_SDA_activations')
                # print(ac)

                saver = tf.train.Saver()
                with tf.Session(graph=self.dha_sda) as sess:
                    sess.run(tf.global_variables_initializer())
                    # writer = tf.summary.FileWriter("demo_class", sess.graph)
                    # writer.close()

                    feed_dicts = {}
                    for i in range(len(self.split_num)):
                        feed_dicts[self.names['V{}'.format(i)]] = self.data_space['data{}'.format(i)]

                    sess.run([self.loss_ae1, self.optimizer_ae1, self.loss_ae2, self.optimizer_ae2,
                              self.loss_ae3, self.optimizer_ae3], feed_dict=feed_dicts)

                    self.loss_list = []
                    self.R_history = []

                    print('Enter train the Space........')
                    t1_ = time.time()
                    feed_batch_dicts = {}
                    for j in range(self.train_step):
                        for batch_i in range(int(np.ceil(len(train_data) / self.batch_size))):
                            for i in range(len(self.split_num)):
                                feed_batch_dicts[self.names['V{}'.format(i)]] = random.choices(self.data_space['data{}'.format(i)],
                                                                                    k=int(self.batch_size/self.fault_num))
                            _ = sess.run(train_op, feed_dict=feed_batch_dicts)
                            loss = sess.run(self.loss_all, feed_dict=feed_batch_dicts)

                        self.loss_list.append(loss)
                        R = sess.run([self.names['R{}'.format(i)] for i in range(len(self.split_num))])  # 半径
                        self.R_history.append(R)
                        print(" Epoch", j, ": loss : ", loss)



                    t2_ = time.time()
                    print('训练时间：%.2f s'% (t2_ - t1_))

                    self.R_list = []
                    self.circle = []
                    self.circle = sess.run([self.names['Cm{}'.format(i)] for i in range(self.fault_num)], feed_dict=feed_dicts)
                    self.R_list = sess.run([self.names['R{}'.format(i)] for i in range(self.fault_num)])

                    w1, w2, w3 = sess.run([self.hidden_w1, self.hidden_w2, self.hidden_w3])
                    b1, b2, b3 = sess.run([self.hidden_b1, self.hidden_b2, self.hidden_b3])
                    self.encoder_weight = [w1, w2, w3]
                    self.encoder_bias = [b1, b2, b3]


                    u_list = sess.run([self.names['U{}'.format(i)] for i in range(self.fault_num)], feed_dict=feed_dicts)
                    DHA_pre = np.zeros(shape=[train_label.shape[0], 1])
                    for i in range(len(self.split_num)): # 每个类的数据
                        var = u_list[i]
                        dis = np.zeros(shape=[var.shape[0], len(self.split_num)])
                        for j in range(len(self.split_num)):  # 和每个球心求距离

                            center = tf.reshape(tf.tile(self.circle[j], [self.split_num[i]]),
                                                [self.split_num[i], len(self.circle[j])])   # tf.tile 对张量进行复制
                            dis_c = sess.run(tf.linalg.norm(var - center, axis=1))
                            dis_c = dis_c / self.R_list[j]
                            dis[:, j] = dis_c

                        tmp_id = self.index_space['index{}'.format(i)]
                        DHA_pre[tmp_id] = np.argmin(dis, axis=1).reshape(-1, 1)

                    DHA_pre = DHA_pre.astype(np.int64)
                    acc = tf.reduce_mean(tf.cast(tf.equal(DHA_pre, train_label), tf.float32))
                    acc = sess.run(acc)

                    # 画r和loss的变化曲线图
                    dir1 = './DHA_SDA_jnu/'
                    if not os.path.exists(dir1):
                        os.mkdir(dir1)
                    picturesdir = './DHA_SDA_jnu/pictures/'
                    if not os.path.exists(picturesdir):
                        os.mkdir(picturesdir)
                    # fig1 = Visualization(train_label, DHA_pre, picturesdir)
                    # fig1.plot_R(self.R_history, self.split_num)
                    # fig1.plot_loss(self.loss_list)

                    saver.save(sess, "./DHA_SDA_jnu/model_" + str(tree_i))
                    # sess.close()
                    return DHA_pre, self.circle, self.R_list, self.encoder_weight, self.encoder_bias, loss, acc


    # -------------------距离计算区----------------
    # 测试数据集 在空间W中到球心的距离
    def test(self, testdata, label, Cm, R, w, b, test=False):

        with tf.Session() as sess:

            hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(testdata, w[0]), b[0]))
            hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, w[1]), b[1]))
            space_test = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, w[2]), b[2]))

            m = space_test.shape[0]
            test_pre = np.zeros(shape=[testdata.shape[0], self.fault_num])
            for i in range(len(self.split_num)):

                center = tf.reshape(tf.tile(Cm[i], [m]), [m, len(Cm[i])])  # tf.tile 对张量进行复制
                distance = sess.run(tf.linalg.norm(space_test - center, axis=1))
                distance /= R[i]
                test_pre[:, i] = distance

            test_pre = np.argmin(test_pre, axis=1).astype(np.int64).reshape(-1, 1)
            acc = tf.reduce_mean(tf.cast(tf.equal(test_pre, label), tf.float32))
            acc = sess.run(acc)

            picturesdir = './DHA_SDA_jnu/pictures/'
            if not os.path.exists(picturesdir):
                os.mkdir(picturesdir)

            acc = accuracy_score(label, test_pre)
            print('accuracy：', acc)
            rec_w = recall_score(label, test_pre, average='macro')
            print('recall score (macro)：', rec_w)
            f1_w = f1_score(label, test_pre, average='macro')
            print('f1_score (macro)：', f1_w)
            precision_w = precision_score(label, test_pre, average='macro',zero_division=1)
            print('precision_score (macro)：', precision_w)

            if test==True:
                evaluation = Visualization(label, test_pre, picturesdir)
                space_test = sess.run(space_test)
                # evaluation.pca_2D(space_test)
                # evaluation.tsne_3d(space_test)
                acc = evaluation.plot_confusion_matrix('DHA_SDA test HAR Confusion Matrix')  # 画混淆矩阵
                return acc, test_pre, space_test

            # evaluation = Visualization(label, test_pre, picturesdir)
            #
            # acc = evaluation.plot_confusion_matrix('DHA_SDA test HAR Confusion Matrix')  # 画混淆矩阵

            # evaluation.plot_roc('DHA_SDA roc')

            # sess.close()
            else:
                return acc,  test_pre




if __name__ == "__main__":

    data = shujuchuli_MFPT.cut_samples()
    train_x, train_y, test_x, test_y = shujuchuli_MFPT.make_datasets(data)
    # train_x = train_x[:200, :]
    # train_y = train_y[:200, :]
    # test_x = test_x[:50, :]
    # test_y = test_y[:50, :]


    '''
    输入、输出的标签均为 0,1,2,3，，，，这种类型的，标签用在求准确率时
    '''

    # fcm = FCM_encoder()
    # fcm_pre = fcm.FCM(encoder_output)
    DHA_classifier = DHA_SDA(300, 150, 75, train_step=400, lr=0.004, fault_num=15)
    dha_pre, circle, R_list, w, b, _, _ = DHA_classifier.fit(train_x, train_y, 1)
    acc, _, _ = DHA_classifier.test(test_x, test_y, circle, R_list, w, b, test=True)
    print(acc)

    # DHA_classifier.plot_R()
    # DHA_classifier.plot_loss()
    # plt.show()



    #
    # data1 = shujuchuli_JNU.cut_samples(-5)
    # train_x1, train_y1, test_x1, test_y1 = shujuchuli_JNU.make_datasets(data1)
    # DHA_classifier1 = DHA_SDA(300, 150, 75, train_step=500, lr=0.004, fault_num=12)
    # dha_pre1, circle1, R_list1, w1, b1, _, _ = DHA_classifier1.fit(train_x1, train_y1, -1)
    # acc1 = DHA_classifier1.test(test_x1, test_y1, circle1, R_list1, w1, b1)
    #
    #
    # data2 = shujuchuli_JNU.cut_samples(-15)
    # train_x2, train_y2, test_x2, test_y2 = shujuchuli_JNU.make_datasets(data2)
    # DHA_classifier2 = DHA_SDA(300, 150, 75, train_step=500, lr=0.004, fault_num=12)
    # dha_pre2, circle2, R_list2, w2, b2, _, _ = DHA_classifier2.fit(train_x2, train_y2, -1)
    # acc2 = DHA_classifier2.test(test_x2, test_y2, circle2, R_list2, w2, b2)
    #
    #
    # data3 = shujuchuli_JNU.cut_samples(-10)
    # train_x3, train_y3, test_x3, test_y3 = shujuchuli_JNU.make_datasets(data3)
    # DHA_classifier3 = DHA_SDA(300, 150, 75, train_step=500, lr=0.004, fault_num=12)
    # dha_pre3, circle3, R_list3, w3, b3, _, _ = DHA_classifier3.fit(train_x3, train_y3, -1)
    # acc1 = DHA_classifier3.test(test_x3, test_y3, circle3, R_list3, w3, b3)
    #
    # data4 = shujuchuli_JNU.cut_samples(5)
    # train_x4, train_y4, test_x4, test_y4 = shujuchuli_JNU.make_datasets(data4)
    # DHA_classifier4 = DHA_SDA(300, 150, 75, train_step=500, lr=0.004, fault_num=12)
    # dha_pre4, circle4, R_list4, w4, b4, _, _ = DHA_classifier4.fit(train_x4, train_y4, -1)
    # acc4 = DHA_classifier4.test(test_x4, test_y4, circle4, R_list4, w4, b4)
    #
    # data5 = shujuchuli_JNU.cut_samples()
    # train_x5, train_y5, test_x5, test_y5 = shujuchuli_JNU.make_datasets(data5)
    # DHA_classifier5 = DHA_SDA(300, 150, 75, train_step=500, lr=0.004, fault_num=12)
    # dha_pre5, circle5, R_list5, w5, b5, _, _ = DHA_classifier5.fit(train_x5, train_y5, -1)
    # acc5 = DHA_classifier5.test(test_x5, test_y5, circle5, R_list5, w5, b5)







