
import mxnet as mx
from mxnet import nd
# from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
# load model
def softmax(x):
    exp = nd.exp(x)
    partition = exp.sum(axis=1,keepdims=True)
    return exp/partition
sym, arg_params, aux_params = mx.model.load_checkpoint("my_mlp",0) # load with net name and epoch num
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=["data"], label_names=[]) # label can be empty
mod.bind(for_training=False, data_shapes=[("data", (1,784))]) # data shape, 1 x 2 vector for one test data record
mod.set_params(arg_params, aux_params)

# X,y  = mnist.train.next_batch(1)
x = scipy.misc.imread("./out2.jpg")

X = nd.array(x).reshape([1,784])
print(x.shape)
pre = mod.predict(X)
pre = nd.softmax(pre[0])

print("predict:",pre)
# print("true:",y)
print(sym)
print("over!")
