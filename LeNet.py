from caffe import layers as L, params as P
import caffe
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import os


caffe_root = '/home/ubuntu/Caffe/caffe/'
os.chdir('/home/ubuntu/Caffe/caffe')
os.system('pwd')

###----------Define the Net Architecture----------###

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1. / 255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()
###------------------------------------------------------------------------###


###---Write the Net Architecture in to prototxt file---###
# with open('mnist/lenet_auto_train.prototxt', 'w') as f:
#     f.write(str(lenet('mnist/mnist_train_lmdb', 64)))
#
#
# with open('mnist/lenet_auto_test.prototxt', 'w') as f:
#     f.write(str(lenet('mnist_test_lmdb', 100)))
###----------------------------------------------------------------###


### SET DEVICE and MODEL ###
caffe.set_device(0)
caffe.set_mode_gpu()
#--------------------------#


# solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)


###---Use SGD optimizer---#
solver = caffe.SGDSolver('examples/mnist/lenet_auto_solver.prototxt')
#--------------------------#


###---test forward speed---###
s = time.time()
solver.net.forward()  # train net
e = time.time()
print  (e-s)    # 0.0746409893036
#----------------------------#


solver.test_nets[0].forward()  # test net (there can be more than one)


###---we use a little trick to tile the first eight images [in one row by transpose(1,0,2)]---###
# plt.imshow(solver.net.blobs['data'].data[0:8, 0].transpose(1, 0, 2).reshape(28, 8*28), cmap='gray')
# print solver.net.blobs['data'].data.shape   #(64,1,28,28)
# print solver.net.blobs['data'].data[0:8, 0].shape   #  (8,28,28)
#----------------------------------------------------------------------#


###--- show 8 images in col ---###
plt.imshow(solver.net.blobs['data'].data[0:8, 0,:,:].reshape(28*8,28),cmap = 'gray')
###----------------------------###


print 'train labels:', solver.net.blobs['label'].data[:8]


###--- show test_net's data source's first 8 image in col ---###
plt.imshow(solver.test_nets[0].blobs['data'].data[:8,0,:,:].reshape(28*8,28),cmap='gray')
###------------------------------------------------------###

print 'test labels:', solver.test_nets[0].blobs['label'].data[:8]


###------------ show all params(W and b) ------------###
# all = np.zeros((20,25*24), dtype=np.float32)
# print all.shape
# solver.step(1)
# for i in range(2):
#     print i, solver.net.params['conv1'][i].data.shape
#     all[:,(i*25):(i*25+25)]=solver.net.params['conv1'][1].diff[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4 * 5, 5 * 5)
#     plt.imshow(all, cmap='gray')
###------------------------------------------------------###



plt.imshow(solver.net.params['conv1'][0].data[:,0,:,:].reshape(4,5,5,5).transpose(0, 2, 1, 3).reshape(20,25),cmap='gray')
# plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4 * 5, 5 * 5), cmap='gray')
plt.imshow(solver.net.params['conv1'][1].data.reshape(2,10), cmap='gray')

print solver.net.params['conv1'][0].data[0,0]       # 5X5 Weights Matrix

# plt.axis('off')
# plt.show()



##---------------------------------------------------##

###------ Let's control the training loop ------###



niter = 200              # Total num in loop
test_interval = 25       # test step

# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))# get integer towards infinity like -1.7 -> -1.0
output = np.zeros((niter, 8, 10))

ms = time.time() # A time-stamp

###---step by step train to know caffe---###

# solver.step(1)
# print solver.net.blobs['loss'].data                   # just one num
#
# solver.test_nets[0].forward(start='conv1')
# print solver.test_nets[0].blobs['score'].data.shape   #(100,10) in solver.prototxt we define test_iter:100, and
#                                                       # in test_net prototxt file data_layer batch_size:100
# output = solver.test_nets[0].blobs['score'].data[:8]  #
# correct = 0
# solver.test_nets[0].forward()   # feed the data to test_net
# correct = solver.test_nets[0].blobs['score'].data.argmax(1)  #argmax() is a num, argmax(0).shape = (10,),argmax(1).shape = (100,) no argmax(2)
#                                                              # so the num in the argmax(num) must be the axis --> 0 mean max_row | 1 max_col
# print correct
###-----------------------------------------------------------------------------------###



###--------- Control the train loop ---------###

for it in range(niter):
    solver.step(1)  #  SGD by Caffe compute forward backward and update w n b for 1 time

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4
me = time.time()
print "time for main loop:", (me-ms)     # about 58.920165062 seconds
###---------------------------------------------------------------###


###--- show the chart of train loss and test accuracy ---###
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))

###-----------------------------------------------------###


###--- we have select 8 batch in test_net and store in output witch shape is ( 200, 8 ,10) ---###
###--- show their data and score ---###
for i in range(8):
    plt.figure(figsize=(2, 2))
    plt.imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    plt.figure(figsize=(10, 2))
    plt.imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    plt.xlabel('iteration')
    plt.ylabel('label')
plt.show()
