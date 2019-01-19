import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import scipy.misc
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
in_units = 784 #输入节点数
h1_units = 300 #隐含层节点数
#************************************************自定义部分开始*************************************************#
#********************参数**********************#
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
b1 = tf.Variable(tf.zeros([h1_units])) #隐含层偏置b1全部初始化为0

W2 = tf.Variable(tf.zeros([h1_units, 10]))   #
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])  #数据输入口
keep_prob = tf.placeholder(tf.float32) #Dropout失活率    ***dropout***
 #******************************************************#

#***************开始画图（计算图）***********#
#定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)    #第一层  
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)   #加入dropout 
hidden2 = tf.matmul(hidden1_drop, W2) + b2   #第二层
y = tf.nn.softmax(hidden2)                   #softmax    ***softmax***

saver = tf.train.Saver()


sess = tf.InteractiveSession()    # 会话窗口
saver.restore(sess,"./model/model.ckpt")
batch_xs, batch_ys = mnist.train.next_batch(1)   #获得要进行训练的数据   获取数据的方式要自己写
tmp_x = batch_xs.reshape([28,28])
print(tmp_x)
scipy.misc.imsave("out3.jpg",tmp_x)
plt.imshow(tmp_x)
print("true label is",batch_ys)
ops = [y]
food = {x:batch_xs,keep_prob:1}
pre = sess.run(y,food)
print("predict is:",pre)
plt.show()


