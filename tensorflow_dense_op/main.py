import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import math
import random
import math
import time
dense_module = tf.load_op_library('cuda_op_kernel.so')
import _dense_grad

LEARNING_RATE = 0.001
BATCH_SIZE = 100
RANGE = 100

print("Tensorflow version: ", tf.__version__)
tf.set_random_seed(42)
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, 1), name="x")

def full_layer(i, units, af, n):
    print("---------full layer", units)
    print("i:", i.get_shape())
    W = tf.get_variable(n+"_W",shape=(i.get_shape()[1], units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    print("W:", W.get_shape())
    b = tf.get_variable(n+"_b",shape=(1,units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    print("b:", b.get_shape())
    #r_ = tf.matmul(i,W)
    #print("r_:", r_.get_shape())
    #r = r_ + b
    r = dense_module.dense(i,W,b)
    if af != None:
        r = af(r)
    return r
 
l = X   

CUSTOM_OP = False
if CUSTOM_OP:
    l =  full_layer(l, 100, tf.nn.relu6, "A")
    l =  full_layer(l, 100, tf.nn.relu6, "B")
    l =  full_layer(l, 100, tf.nn.relu6, "C")
    l =  full_layer(l, 100, tf.nn.relu6, "D")
    l =  full_layer(l, 1, None, "E")
else:
    l =  tf.layers.dense(l, units=100, activation=tf.nn.relu6)
    l =  tf.layers.dense(l, units=100, activation=tf.nn.relu6)
    l =  tf.layers.dense(l, units=100, activation=tf.nn.relu6)
    l =  tf.layers.dense(l, units=100, activation=tf.nn.relu6)
    l =  tf.layers.dense(l, units=1, activation=None)
print(l)

response = l 
           
Y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

loss = tf.reduce_mean(tf.pow((response - Y),2))
    
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

session = tf.Session()        
session.run(tf.global_variables_initializer())

def blackbox_function(x):
    return 1.0+math.sin(2*x*math.pi*2/RANGE)*0.5 + x/60.0

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim([-3,3])
plt.xlim([-RANGE,RANGE])

line_target, = ax.plot([0], [1], 'g-')
line_nn, = ax.plot([0], [1], 'r-')
d_x = [[(random.random()*2-1)*RANGE] for _ in range(BATCH_SIZE) ]
d_y = [[blackbox_function(x[0])] for x in d_x]
scatter = ax.scatter([d_x], [d_y], c='blue', s=4)

def itemize(d):
    return list(map(lambda _: [_],d))

starttime = time.time()
iteration = 0
while True:
    d_x = [[(random.random()*2-1)*RANGE] for _ in range(BATCH_SIZE) ]
    d_y = [[blackbox_function(x[0])] for x in d_x]



    #print("Training step begin")
    [tr, l, r] = session.run(fetches=[train_step, loss, response],feed_dict={X: d_x, Y: d_y})
    print("Iteration: %d Loss: %f" % (iteration, l))
    iteration += 1
    #print("Training step end")
    
    
    dp_x = [x*2*RANGE/100.0 - RANGE for x in range(100)]
    dp_x_ = [[x] for x in dp_x]
    
    #print("Evaluation step begin")
    [r] = session.run(fetches=[response],feed_dict={X: dp_x_})
    #print("Evaluation step end")
    
    dp_target = [ blackbox_function(_) for _ in dp_x]
    dp_nn = [_[0] for _ in r]

    line_target.set_data(dp_x,dp_target)
    line_nn.set_data(dp_x,dp_nn)

    sp = []
    for i in range(len(d_x)):
        sp.append([d_x[i][0], d_y[i][0]])

    scatter.set_offsets(sp)
    

    fig.canvas.draw()
    if iteration > 1000:
        break
print("%f ms per iteration" % (((time.time() - starttime)*1000)/(iteration)))
