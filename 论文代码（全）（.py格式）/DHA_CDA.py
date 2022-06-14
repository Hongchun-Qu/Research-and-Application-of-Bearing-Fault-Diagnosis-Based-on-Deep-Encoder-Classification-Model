import tensorflow as tf
import numpy as np
import itertools
import time
from paint_figure import Visualization
import shujuchuli
import os
import shujuchuli_MFPT
import random
from sklearn.metrics import *
import shujuchuli_JNU

'''
理论参考：https://blog.csdn.net/haitao111313/article/details/7826915
论文：动态特征空间的超球体分类算法
DHA代码来自：
https://github.com/crazy3water/Dynamic-Hypersphere-Algorithm-for-Classification
'''

class DHA_CDA():
    def __init__(self, unit1, unit2, unit3, model_name='dha_cda', train_step=10, fault_num=10,
                 batch_size=500, lr=0.8, R_constant=1.0, P=1.0, P_R2Cm=1.0, P_class=0.0001):
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
        self.delta1 = 1e-3  # 雅克比惩罚项系数

        self.unit1 = unit1
        self.unit2 = unit2
        self.unit3 = unit3
        self.batch_size = batch_size
        self.pop = np.array([600, self.unit1, self.unit2, self.unit3, 1])
        self.model_name = model_name
        self.dha_cda = tf.Graph()
        self.fault_num = fault_num

        self.data_space = {}
        self.index_space = {}
        self.names = {}
        self.split_num = []  # 每个类别的数据的个数

        # lrp
        self.weights = []
        self.activations = []


    # 编码器的损失函数
    def encoder_loss(self, data, outputdata, hidden_w, hidden):

        error_loss = tf.reduce_mean(tf.square(outputdata - data))
        # 雅克比矩阵 https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
        J_w = tf.transpose(hidden_w)
        d_sigmoid = hidden * (1 - hidden)
        contractive = self.delta1 * tf.reduce_sum(d_sigmoid ** 2 * tf.reduce_sum(J_w ** 2, axis=1), axis=1)

        loss_ae = error_loss + contractive
        optimizer_ae = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_ae, var_list=tf.trainable_variables())

        return loss_ae, optimizer_ae


    # 在圆心外惩罚
    # tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
    # 小于min的让它等于min，大于max的元素的值等于max。
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
        with self.dha_cda.as_default():
            with tf.name_scope(self.model_name):

                with tf.name_scope('loss_pow'):
                    # len(self.split_num) = 3
                    split_list = np.arange(0, len(self.split_num))
                    for i in range(len(self.split_num)):
                        with tf.name_scope('loss_pow{}'.format(i)):
                            split_list_ = np.delete(split_list, i)  # 按行删除，删除第i+1个数，因为从0开始
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

                            # 如果 半径 Ri 小于 0 的话，就令Rn = Ri, 否则的话，令Rn = 0
                            Rn = tf.where(tf.greater(np.float64(0), self.names['R{}'.format(i)]),
                                          self.names['R{}'.format(i)], 0)
                            # loss = 该类中所有样本的欧式距离 + P * { 本类样本 + 非本类样本 }
                            # P: 每个球体的惩罚系数 = 1
                            self.names['loss{}_pow'.format(i)] = tf.pow(g1n1, 2) + tf.pow(g2n1, 2) + tf.pow(Rn, 2)

                    loss_pow = 0
                    for i in range(len(self.split_num)):
                        loss_pow = loss_pow + self.names['loss{}_pow'.format(i)]

                        # tf.greater(a,b) 功能：通过比较a、b两个值的大小来输出对错。
                        # 比如：当a=4，b=3时，输出结果为：true；当a=2，b=3时，输出结果为：false。

                        # tf.where的作用是根据condition,返回相对应的x或y,返回值是一个tf.bool类型的Tensor
                        # 比如：A =tf.where(False,123,321) 输出：A = Tensor("Select:0", shape=(), dtype=int32)
                        # 然后sess.run(A) 可得 A = 321
                        # sess.run(tf.where(True,123,321)) = 123

                # tolist():转换为list
                # itertools.combinations：
                # 创建一个迭代器，返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序(不带重复)
                with tf.name_scope('lossR2Cm'):
                    combine = list(itertools.combinations(np.arange(0, len(self.split_num)).tolist(), 2))
                    loss_R2Cm = 0
                    for i in combine:
                        with tf.name_scope('lossR2Cm{}{}'.format(i[0], i[1])):
                            # Cm_normal：两个球的半径的和 - 两个球的中心差的线性范数
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
        with self.dha_cda.as_default():
            with tf.name_scope(self.model_name):

                for class_i in range(self.fault_num):
                    i_d = np.argwhere(train_label == class_i)
                    i_d = i_d[:, 0]
                    self.split_num.append(len(i_d))
                    self.data_space['data{}'.format(class_i)] = train_data[i_d]
                    self.index_space['index{}'.format(class_i)] = i_d  # 二维

                # names的类型是dict：3，即建立一个字典用来存输入的数据 ‘V0’：shape=(?,13)
                with tf.name_scope('InitVariable'):
                    # 动态变量名设置
                    for i in range(len(self.split_num)):
                        # self.split_num是一个列表：元素为每一类的数据个数 ，即  [52, 63, 45]
                        # len(self.split_num)为类别数
                        with tf.name_scope('V{}'.format(i)):  # self.x_w = data.shape[1] 即列数
                            self.names['V{}'.format(i)] = tf.placeholder(
                                tf.float64, shape=[None, train_data.shape[1]], name='Input{}'.format(i))


                    with tf.name_scope('weight'):
                        self.hidden_w1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[0], self.pop[1]]),
                                                        trainable=True), tf.float64, name='hidden1_w')  # float32
                        output_w1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[1], self.pop[0]]),
                                                        trainable=True), tf.float64, name='output1_w')
                        self.hidden_w2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[1], self.pop[2]]),
                                                        trainable=True), tf.float64, name='hidden2_w')  # float32
                        output_w2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[2], self.pop[1]]),
                                                        trainable=True), tf.float64, name='output2_w')
                        self.hidden_w3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[2], self.pop[3]]),
                                                        trainable=True), tf.float64, name='hidden3_w')  # float32
                        out_w3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[3], self.pop[2]]),
                                                     trainable=True), tf.float64, name='out3_w')

                        self.weights = [self.hidden_w1, self.hidden_w2, self.hidden_w3]


                    with tf.name_scope('bias'):
                        self.hidden_b1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[1]]),
                                                        trainable=True), tf.float64, name='hidden1_b')
                        output_b1 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[0]]),
                                                        trainable=True), tf.float64, name='output1_b')
                        self.hidden_b2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[2]]),
                                                        trainable=True), tf.float64, name='hidden2_b')
                        output_b2 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[1]]),
                                                        trainable=True), tf.float64, name='output2_b')
                        self.hidden_b3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[3]]),
                                                        trainable=True), tf.float64, name='hidden3_b')
                        out_b3 = tf.cast(tf.Variable(initial_value=tf.random_normal(shape=[self.pop[4], self.pop[2]]),
                                                     trainable=True), tf.float64, name='out3_b')


                    # R_constant:初始化半径
                    # 每个类的球的半径
                    with tf.name_scope('R'):
                        for i in range(len(self.split_num)):
                            self.names['R{}'.format(i)] = tf.Variable(initial_value=self.R_constant,
                                                                 dtype=tf.float64, name='R{}'.format(i), trainable=True)

                with tf.name_scope('encoder_layer'):

                    for i in range(len(self.split_num)):

                        hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(self.names['V{}'.format(i)], self.hidden_w1), self.hidden_b1))
                        output_ae1 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, output_w1), output_b1))
                        hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, self.hidden_w2), self.hidden_b2))
                        output_ae2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, output_w2), output_b2))
                        self.names['U{}'.format(i)] = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, self.hidden_w3), self.hidden_b3))
                        out_ae3 = tf.nn.sigmoid(tf.add(tf.matmul(self.names['U{}'.format(i)], out_w3), out_b3))

                        # 逐层训练
                        self.loss_cae1, self.optimizer_cae1 = self.encoder_loss(
                            self.names['V{}'.format(i)], output_ae1, self.hidden_w1, hidden1)

                        self.loss_cae2, self.optimizer_cae2 = self.encoder_loss(
                            hidden1, output_ae2, self.hidden_w2, hidden2)

                        self.loss_cae3, self.optimizer_cae3 = self.encoder_loss(
                            hidden2, out_ae3, self.hidden_w3, self.names['U{}'.format(i)])

                    self.activations = [self.names['V{}'.format(i)], hidden1, hidden2, self.names['U{}'.format(i)]]

                # 球心
                # 对每个类的 每个 layer 求均值
                # axis=0，表示对第一维度（行）减少，减少行的方法是对所有列求平均，即在行上压缩减少为一行。
                with tf.name_scope('circle'):
                    for i in range(len(self.split_num)):
                        self.names['Cm{}'.format(i)] = tf.reduce_mean(self.names['U{}'.format(i)], axis=0)


    def fit(self, train_data, train_label, tree_i):
        '''
        :param train_data: 训练数据集
        :param train_label: 训练集的标签，输入的标签应该是一位的，即1,2, 3......
        :return:
        '''
        with self.dha_cda.as_default():
            with tf.name_scope(self.model_name):

                self.gen_model(train_data, train_label)
                dha_loss = self.dha_loss()

                with tf.name_scope('Optimizer'):
                    learning_rate = tf.train.exponential_decay(self.lr, self.train_step, decay_steps=100,
                                                               decay_rate=0.8)
                    train_op = tf.train.AdamOptimizer(learning_rate).minimize(dha_loss)

                for weight in self.weights:
                    tf.add_to_collection('LRP_CDA_weights'+str(tree_i), weight)

                for act in self.activations:
                    tf.add_to_collection('LRP_CDA_activations'+str(tree_i), act)

                saver = tf.train.Saver()
                with tf.Session(graph=self.dha_cda) as sess:
                    sess.run(tf.global_variables_initializer())
                    # writer = tf.summary.FileWriter("demo_class", sess.graph)
                    # writer.close()

                    feed_dicts = {}
                    for i in range(len(self.split_num)):
                        feed_dicts[self.names['V{}'.format(i)]] = self.data_space['data{}'.format(i)]

                    sess.run([self.loss_cae1, self.optimizer_cae1, self.loss_cae2,self.optimizer_cae2,
                              self.loss_cae3, self.optimizer_cae3], feed_dict=feed_dicts)

                    self.loss_list = []
                    self.R_history = []

                    print('Enter train the Space........')
                    t1_ = time.time()
                    feed_batch_dicts = {}
                    for j in range(self.train_step):
                        for batch_i in range(int(np.ceil(len(train_data) / self.batch_size))):
                            for i in range(len(self.split_num)):
                                feed_batch_dicts[self.names['V{}'.format(i)]] = random.choices(
                                    self.data_space['data{}'.format(i)],
                                    k=int(self.batch_size/self.fault_num))
                            _ = sess.run(train_op, feed_dict=feed_batch_dicts)
                            loss = sess.run(self.loss_all, feed_dict=feed_batch_dicts)

                        self.loss_list.append(loss)
                        R = sess.run([self.names['R{}'.format(i)] for i in range(len(self.split_num))])  # 半径
                        self.R_history.append(R)
                        print(" Epoch", j, ": loss : ", loss)


                    t2_ = time.time()
                    print('训练时间：%.2f s'% (t2_ - t1_))

                    self.R_list = []   # 半径
                    self.circle = []    # 球心
                    self.circle = sess.run([self.names['Cm{}'.format(i)] for i in range(self.fault_num)], feed_dict=feed_dicts)
                    self.R_list = sess.run([self.names['R{}'.format(i)] for i in range(self.fault_num)])

                    w1, w2, w3 = sess.run([self.hidden_w1, self.hidden_w2, self.hidden_w3])
                    b1, b2, b3 = sess.run([self.hidden_b1, self.hidden_b2, self.hidden_b3])
                    self.encoder_weight = [w1, w2, w3]
                    self.encoder_bias = [b1, b2, b3]


                    u_list = sess.run([self.names['U{}'.format(i)] for i in range(self.fault_num)], feed_dict=feed_dicts)
                    DHA_pre = np.zeros(shape=[train_label.shape[0], 1])
                    for i in range(len(self.split_num)):   # 每个类的数据
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
                    dir1 = './DHA_CDA_jnu/'
                    if not os.path.exists(dir1):
                        os.mkdir(dir1)
                    picturesdir = './DHA_CDA_jnu/pictures/'
                    if not os.path.exists(picturesdir):
                        os.mkdir(picturesdir)
                    # fig1 = Visualization(train_label, DHA_pre, picturesdir)
                    # fig1.plot_R(self.R_history, self.split_num)
                    # fig1.plot_loss(self.loss_list)
                    saver.save(sess, "./DHA_CDA_jnu/model_" + str(tree_i))  # 用于LRP
                    sess.close()

                    return DHA_pre, self.circle, self.R_list, self.encoder_weight, self.encoder_bias, loss, acc


    # -------------------距离计算区----------------
    # 测试数据集 在空间W中到球心的距离
    def test(self, testdata, test_label, Cm, R, w, b, test=False):

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
            acc = tf.reduce_mean(tf.cast(tf.equal(test_pre, test_label), tf.float32))
            acc = sess.run(acc)
            print(acc)

            picturesdir = './DHA_CDA_jnu/pictures/'

            acc = accuracy_score(test_label, test_pre)
            print('accuracy：', acc)
            rec_w = recall_score(test_label, test_pre, average='macro')
            print('recall score (macro)：', rec_w)
            f1_w = f1_score(test_label, test_pre, average='macro')
            print('f1_score (macro)：', f1_w)
            precision_w = precision_score(test_label, test_pre, average='macro')
            print('precision_score (macro)：', precision_w)

            if test == True:
                fig2 = Visualization(test_label, test_pre, picturesdir)
                acc = fig2.plot_confusion_matrix('DHA_CDA test HAR Confusion Matrix')  # 画混淆矩阵
                space_test = sess.run(space_test)
                fig2.pca_2D(space_test)
                fig2.tsne_3d(space_test)
                return acc, test_pre, space_test

                # fig2.pca_encoder_2D(space_test)

            # fig2 = Visualization(test_label, test_pre, picturesdir)
            # acc = fig2.plot_confusion_matrix('DHA_CDA test HAR Confusion Matrix')  # 画混淆矩阵

            # sess.close()
            else:
                return acc, test_pre





if __name__ == "__main__":

    data = shujuchuli_JNU.cut_samples(-10)
    train_x, train_y, test_x, test_y = shujuchuli_JNU.make_datasets(data)
    # train_x = train_x[:150, :]
    # train_y = train_y[:150, :]
    # val_x = val_x[:50, :]
    # val_y = val_y[:50, :]

    DHA_classifier = DHA_CDA(300, 150, 75, train_step=600, lr=0.006, fault_num=12)
    dha_pre, circle, R_list, w, b, _, _ = DHA_classifier.fit(train_x, train_y, 1)
    acc = DHA_classifier.test(test_x, test_y, circle, R_list, w, b, test=True)


    #
    # data1 = shujuchuli_JNU.cut_samples(-5)
    # train_x1, train_y1, test_x1, test_y1 = shujuchuli_JNU.make_datasets(data1)
    # DHA_classifier1 = DHA_CDA(300, 150, 75, train_step=500, lr=0.006, fault_num=12)
    # dha_pre1, circle1, R_list1, w1, b1, _, _ = DHA_classifier1.fit(train_x1, train_y1, -1)
    # acc1 = DHA_classifier1.test(test_x1, test_y1, circle1, R_list1, w1, b1)
    #
    #
    # data2 = shujuchuli_JNU.cut_samples(-15)
    # train_x2, train_y2, test_x2, test_y2 = shujuchuli_JNU.make_datasets(data2)
    # DHA_classifier2 = DHA_CDA(300, 150, 75, train_step=500, lr=0.006, fault_num=12)
    # dha_pre2, circle2, R_list2, w2, b2, _, _ = DHA_classifier2.fit(train_x2, train_y2, -1)
    # acc2 = DHA_classifier2.test(test_x2, test_y2, circle2, R_list2, w2, b2)
    #
    #
    # data3 = shujuchuli_JNU.cut_samples(0)
    # train_x3, train_y3, test_x3, test_y3 = shujuchuli_JNU.make_datasets(data3)
    # DHA_classifier3 = DHA_CDA(300, 150, 75, train_step=500, lr=0.006, fault_num=12)
    # dha_pre3, circle3, R_list3, w3, b3, _, _ = DHA_classifier3.fit(train_x3, train_y3, -1)
    # acc1 = DHA_classifier3.test(test_x3, test_y3, circle3, R_list3, w3, b3)
    #
    # data4 = shujuchuli_JNU.cut_samples(5)
    # train_x4, train_y4, test_x4, test_y4 = shujuchuli_JNU.make_datasets(data4)
    # DHA_classifier4 = DHA_CDA(300, 150, 75, train_step=500, lr=0.006, fault_num=12)
    # dha_pre4, circle4, R_list4, w4, b4, _, _ = DHA_classifier4.fit(train_x4, train_y4, -1)
    # acc4 = DHA_classifier4.test(test_x4, test_y4, circle4, R_list4, w4, b4)
    #
    # data5 = shujuchuli_JNU.cut_samples()
    # train_x5, train_y5, test_x5, test_y5 = shujuchuli_JNU.make_datasets(data5)
    # DHA_classifier5 = DHA_CDA(300, 150, 75, train_step=500, lr=0.006, fault_num=12)
    # dha_pre5, circle5, R_list5, w5, b5, _, _ = DHA_classifier5.fit(train_x5, train_y5, -1)
    # acc5 = DHA_classifier5.test(test_x5, test_y5, circle5, R_list5, w5, b5)









