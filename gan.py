import matplotlib.pyplot as plt
import numpy as np
import theano
#from numpy import im
from theano import tensor as T
import pickle
import gzip
from PIL import Image
import lasagne
from lasagne import layers as l
from lasagne.layers import batch_norm
from lasagne.nonlinearities import rectify as reLU
from lasagne.nonlinearities import softmax as softmax
exception_verbosity='high'

def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

def load_data_shared(filename = "../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data[0], validation_data[0], test_data[0])
    #return [shared(training_data[0]), shared(validation_data[0]), shared(test_data[0])]

tr, val,test = load_data_shared( "/Users/anastasia/Downloads/neural-networks-and-deep-learning-master-3/data/mnist.pkl.gz")
tr = np.reshape(tr, tr.shape + (1,1))

"""
i = 255 * tr[10]
i1 = np.reshape(i,(28,28)).astype("uint8")
i1 = i1
im = Image.fromarray(i1)
im.show()
im.save('test1.png')
#i = i.astype("uint8")

"""
def save_imgs(arr):
    for i in range((arr[0]).size):
        img = (np.uint8)(255 * arr[i])
        img1 = Image.fromarray(img)
        img1.show()
        img1.save('generated_result_i.png')
    print('images saved')


def make_noise_sample(b,w,h):
    n = np.random.random_sample([b,w,h])
    return np.reshape(n, n.shape + (1,))

noise_W = 4
noise_H = 4


def build_generator(input_var = None):
    #D_inp = T.tensor4('ds')
    G = l.InputLayer(shape=(None, 1, noise_H, noise_W), input_var = input_var)
    G = batch_norm(l.DenseLayer(G,num_units = (noise_H * noise_W * 256) ,nonlinearity= reLU))
    G = l.ReshapeLayer(G, shape=([0], 256, noise_H, noise_W))#4
    G = l.TransposedConv2DLayer(G,1,filter_size=(2,2), stride=(2,2), output_size=8)#8
    G = batch_norm(l.Conv2DLayer(G, 40, (3,3), nonlinearity=reLU, pad='full'))#10
    G = l.TransposedConv2DLayer(G,1,filter_size=(2,2), stride=(2,2),output_size=20 )#20
    G = batch_norm(l.Conv2DLayer(G, 20, (3,3),nonlinearity=reLU, pad = 'full'))#22
    G = batch_norm(l.Conv2DLayer(G,20,(5,5), nonlinearity=reLU, pad = 'full'))#26
    G = batch_norm(l.Conv2DLayer(G,1,(3,3), nonlinearity=reLU, pad = 'full'))#28

    return G


def build_discriminator(input_var = None):
    #D_inp = T.tensor4('Ds')
    D = l.InputLayer(shape=(None, 1, 28,28), input_var = input_var)
    D = l.Conv2DLayer(D, num_filters= 20, filter_size=(5,5),nonlinearity= reLU, W=lasagne.init.GlorotUniform())
    #D = l.Conv2DLayer(D,1,filter_size=(2,2), stride=2, nonlinearity=reLU)
    D = l.DropoutLayer(D, p=0.2)
    D = l.Conv2DLayer(D, num_filters= 20, filter_size=(5,5),nonlinearity= reLU, W=lasagne.init.GlorotUniform())
    #D = l.Conv2DLayer(D,1,filter_size=(2,2), stride=2, nonlinearity=reLU)
    D = l.DropoutLayer(D, p=0.2)
    D = l.DenseLayer(l.dropout(D, p = 0.5), num_units= 256, nonlinearity= reLU)
    D = l.DenseLayer(l.dropout(D, p = 0.5), num_units= 1, nonlinearity= softmax)


    #D1.params = D.params
    return D


input_var = T.tensor4('inp')
input_var1 = T.tensor4('input1')
input_var2 = T.tensor4('input2')

gen = build_generator(input_var)
d1 = build_discriminator(input_var1)#datas
d2 = build_discriminator(l.get_output(gen))#noise
d2.params = d1.params

loss_d = (T.log(l.get_output(d1)) + T.log(1 - (l.get_output(d2)))).mean()
loss_g = (T.log(l.get_output(d2))).mean()

updates_d =lasagne.updates.sgd(loss_d,l.get_all_params(d1, trainable = True),learning_rate=0.1)
updates_g =lasagne.updates.sgd(loss_g,l.get_all_params(gen, trainable = True),learning_rate=0.1)

generated  = l.get_output(gen)
show_gen = theano.function([input_var], generated)

g_train = theano.function([input_var], loss_g, updates=updates_g)
d_train = theano.function([input_var1, input_var], loss_d, updates=updates_d)


def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]



def train(mini_batch_size,k, epoch):
    for i in range(epoch):
        print('epoch ', i ,'started' )
        for j in range(k):

            for batch in iterate_minibatches(tr, 10, shuffle=True):
                noise_batch = make_noise_sample(mini_batch_size, noise_W,noise_H)
                #batch = T.tensor(None, mini_batch_size,28,28)
                inp = np.reshape(batch,(mini_batch_size,1,28,28))
                input1 = show_gen(noise_batch)
                d_train(inp,input1)

                #d_train(input1)

        noise_batch1 = make_noise_sample(mini_batch_size, noise_W,noise_H)
        input = noise_batch1
        g_train(input)
        #save generated samples
        print("epoch ", i, "finished")
        #generate some samples
        n_s = make_noise_sample(mini_batch_size,noise_W,noise_H)
        imgs = show_gen(n_s)
        save_imgs(imgs)



def save_model():
    params = [gen.params, d1.params]
    file = open("/Users/anastasia/neural-networks-and-deep-learning/data/saved_model.save", "wb")
    pickle.dump(params,file,protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    print("params saved")

train(10,1,100)
save_model()


