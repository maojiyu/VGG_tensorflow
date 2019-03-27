import os
import numpy as np
import tensorflow as tf
import csv
from VGG_TensorFlow.tensorflow_vgg import vgg16
from scipy.misc import imread, imresize
from VGG_TensorFlow.tensorflow_vgg import utils

codes = None
data_dir = './/flower_photos/'
contents = os.listdir(data_dir)  # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
classes = [each for each in contents if os.path.isdir(data_dir + each)]  # 判断某一路径是否为文件路径
###########################训练网络##########################################

# 我们在得到数据集的特征以后，我们又设计使用了一个256维的全连接层，一个5维的全连接层(因为数据集中给出的种类是五种不同种类的花朵)，和一
# 个softmax层，这里即你微调参数发挥的地方，网络结构可以任意修改，可以不断尝试其他的结构以找到最适合的结构

# 输入数据的维度
inputs_ = tf.placeholder(tf.float32, shape=[None, 4096])
# 标签数据的维度
labels_ = tf.placeholder(tf.int64, shape=[None, 5])
# 加入一个256维的全连接的层
fc = tf.contrib.layers.fully_connected(inputs_, 256)
# 加入一个5维的全连接层
logits = tf.contrib.layers.fully_connected(fc, 5, activation_fn=None)
# 计算cross entropy的值
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
# 计算损失函数
cost = tf.reduce_mean(cross_entropy)
# 采用用的最广泛的AdamOptimizer优化器
optimizer = tf.train.AdamOptimizer().minimize(cost)
# 得到最后的预测分布
predicted = tf.nn.softmax(logits)
# 计算准确度
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 保存模型的保存器
saver = tf.train.Saver()
with tf.Session() as sess:
    # 构建VGG16模型对象
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [1, 224, 224, 3])
    with tf.name_scope('content_vgg'):
        # 载入VGG16模型
        vgg.build(input_)
        # 对不同种类的花分别用VGG16计算特征值
        # 以下得到一个codes数组和一个labels数组，分别存储了所有花朵的特征值和类别
        # 载入图片并放入batch数组中
        img = utils.load_image('.//test_data/Dandelion.jpg')
        img1=img.reshape((1, 224, 224, 3))
        # 如果图片数量到了batch_size则开始具体的运算
        feed_dict = {input_: img1}
        # 计算特征值
        codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)  # 到这里codes_batch的值为[n,4096]
        # 将结果放入到codes数组中
        codes = codes_batch
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        feed = {inputs_: codes}
        prob = sess.run(predicted, feed_dict=feed)[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(classes[p], prob[p])
