# -*- coding utf-8 -*-
# C:\Users\lxg\Documents\Python
# Author:李小根
# Time:2019/2/28
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 定义命令行参数
# tf.flags.DEFINE_integer("max_step", 100, "训练模型参数")  # [名字， 默认值， 说明]
# tf.flags.DEFINE_string("model_dir", " ", "模型文件的加载路径")

# 定义获取命令行参数名字
# FLAGS = tf.flags.FLAGS

mnist = input_data.read_data_sets('./MNIST_DNN/data/', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)  # 28*28
y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch, in_height, in_width, in_channels]


# 初始化权重于定义网络结构
# 构建2个卷积层和2个池化层，1个全连接层和1个输出层的卷积神经网络
def init_weight(shape):
    # with tf.variable_scope("weight"):
    weight = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(weight)


def init_bias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


# 卷积层
def conv2d(x, w):
    conv_2d = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    return conv_2d


# 池化层
def max_pool(x):
    max_pool_2d = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return max_pool_2d


def model(x, w1, w2, w3, w4, b1, b2, b3, b4, p_keep_conv, p_keep_hidden):
    # 第一组卷积层及池化层,最后dropout一些神经元
    conv1 = tf.nn.relu(conv2d(x, w1) + b1)
    pool1 = max_pool(conv1)
    l1 = tf.nn.dropout(pool1, p_keep_conv)

    # 第二组卷积层及池化层,最后dropout一些神经元
    # conv2 = tf.nn.relu(conv2d(l1, w2) + b2)
    # pool2 = max_pool(conv2)
    # l2 = tf.nn.dropout(pool2, p_keep_conv)

    # 第二组卷积层及池化层,最后dropout一些神经元
    conv2 = tf.nn.relu(conv2d(l1, w2) + b2)
    pool2 = max_pool(conv2)
    l2 = tf.reshape(pool2, [-1, 64 * 7 * 7])  # 扁平化处理成一维[-1, 128*4*4]
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # 全连接层,最后dropout一些神经元
    l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
    l3 = tf.nn.dropout(l3, p_keep_hidden)

    # 输出层
    y = tf.matmul(l3, w4) + b4
    return y  # 返回预测值


# 设置卷积核大小为3×3
w1 = init_weight([5, 5, 1, 32])  # [width, height, in_channels, out_channels]
b1 = init_bias([32])
w2 = init_weight([5, 5, 32, 64])  # 64个卷积核从32个平面抽取64个特征平面
b2 = init_bias([64])

w3 = init_weight([64 * 7 * 7, 1024])  # 全连接层,上一层有128*4*4个神经元,全连接层有625个神经元
b3 = init_bias([1024])
w4 = init_weight([1024, 10])  # 输出层
b4 = init_bias([10])

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)
y = model(x_image, w1, w2, w3, w4, b1, b2, b3, b4, p_keep_conv, p_keep_hidden)

# 定义损失函数
# with tf.variable_scope("loss"):
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# with tf.variable_scope("accuracy"):
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 收集tensor
# tf.tmp.scalar("losses", loss)
# tf.tmp.histogrm('weights', weight)

# 合并tensor的op
# merged = tf.tmp.merge_all()

# 保存模型
# saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # writeFile = tf.tmp.FileWriter("./tmp/tmp/", sess.graph)  建立事件文件

    # if os.path.exists("./tmp/ckpt/checkpoint"):
    #   # 加载模型
    #   saver.restore(sess, "./tmp/ckpt/model")

    for i in range(200):  # FLAGS.max_step
        for batch in range(n_batch):
            xs, ys = mnist.train.next_batch(batch)
            sess.run(train_step, feed_dict={x: xs, y_: ys, p_keep_conv: 0.8, p_keep_hidden: 0.5})

        # tmp = sess.run(merged)
        # writeFile.add_summary(merged, i)
        # if i % 10 == 0:
            # saver.save(sess, './tmp/ckpt/model')
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels,
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})
        print('Iter:' + str(i) + "\t" + 'Test accuracy:' + str(acc))

