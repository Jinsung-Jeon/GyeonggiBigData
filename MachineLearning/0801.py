# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf

#실수형으로 표현하기 위해 마침표 사용
matrix1 = tf.constant([[3., 3.]])   # dim(matrix1) = (1,2)
matrix2 = tf.constant([[2.],[2.]])  # dim(matrix2) = (2,1)

product = tf.matmul(matrix1, matrix2)

sess = tf.Session() #그래프그려주기 위한 tf.Session
result = sess.run(product)
print(result)
sess.close()

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  print(sess.run(state))

  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

#### tensorflow를 이용한 선형회귀 ####
W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b

#loss
loss = tf.reduce_sum(tf.square(linear_model - y)) #sse 
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) #0.01은 학습률
train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
#training Loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:x_train,y:y_train})
    
# evaluate raining accuracy
sess.run([W,b,loss],{x:x_train, y:y_train})
curr_W, curr_b, curr_loss = sess.run([W,b,loss],{x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
sess.close()

#### 변수가 3개인 션형회귀 ####
import tensorflow as tf
import math
x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable([1.], name = 'weight1')
w2 = tf.Variable([1.], name = 'weight2')
w3 = tf.Variable([1.], name = 'weight3')
b = tf.Variable([1.], name = 'bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

#loss
cost = tf.reduce_mean(tf.square(hypothesis - Y)) #mse사용
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) #gradientdescent 변경가능
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
    sess.run(train, {x1:x1_data,x2:x2_data,x3:x3_data,Y:y_data})
    sess.run([w1,w2,w3,b,cost],{x1:x1_data,x2:x2_data,x3:x3_data,Y:y_data})
    curr_W1,curr_W2,curr_W3, curr_b, curr_loss = sess.run([w1,w2,w3,b,cost],{x1:x1_data,x2:x2_data,x3:x3_data,Y:y_data})
    print("W1: %s W2: %s W3: %s b: %s loss: %s"%(curr_W1,curr_W2,curr_W3, curr_b, curr_loss))
    
    
    


