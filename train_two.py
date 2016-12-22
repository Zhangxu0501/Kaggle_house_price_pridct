#coding:utf-8
import tensorflow as tf
import numpy as np
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def process(i, data):
    info = i.split(',')
    info[-1] = info[-1].replace('\n', '')
    info = info[1:]
    for j in range(0, len(info)):
        info[j] = get_num(info[j], data[j])
    return info[-1], info[0:-1]


def get_num(info_j, data_k):
    if data_k == True:
        if info_j == 'NA':
            return 0.0
        return float(info_j)
    else:
        if data_k.has_key(info_j):
            return info_j
        else:
            data_k[info_j] = len(data_k) - 1
            return info_j


def get_data(i):
    info = i.split(',')
    info[len(info) - 1] = info[len(info) - 1].replace('\n', '')
    info_i = info[1:]
    data = list()
    for i in info_i:
        try:
            float(i)
            data.append(True)
        except:
            data.append({i: -1})
    return data


def get_cv(house_info):
    new_house = list()
    for house in house_info:
        now = list()
        for j in range(len(house)):
            for k in range(j, len(house)):
                now.append(house[j] * house[k])
        new_house.append(now)
    return new_house


file = open("/Users/zhangxu/Desktop/train.csv", 'r')
lines = file.readlines()
lines = lines[1:]
data = get_data(lines[0])

house_info = list()
price_info = list()
for i in range(0, len(lines)):
    y, x = process(lines[i], data)
    house_info.append(x)
    price_info.append(y)

for i in range(len(lines)):
    plus = 0
    j = 0
    print len(data), len(house_info[0])
    while(j < len(data)):
        if(data[j] == True):
            j += 1
            continue
        l = list()
        for k in range(len(data[j])-1):
            l.append(0)
        if(data[j][house_info[i][j + plus]] != -1):
            l[data[j][house_info[i][j + plus]]] = 1
        house_info[i] = house_info[i][:j + plus] + l + house_info[i][j + plus + 1:]
        j += 1
        plus += len(l) - 1

house_info = np.float32(house_info)
price_info = np.float32(price_info)
price_info.resize((1460, 1))

house_info_cv = get_cv(house_info)
house_info = np.hstack((house_info, house_info_cv))

print price_info.shape
print house_info.shape
#house_info_cv=get_cv(house_info)
#house_info_new=np.hstack((house_info,house_info_cv))


#以上是对特征的 一系列处理,字符串转为float,将离散的特征变为数字
# 处理之后得到 N*3239的矩阵,开始训练


x=tf.placeholder(dtype=tf.float32,shape=[None,34452])
y=tf.placeholder(dtype=tf.float32,shape=[None,1])


xx_normal=tf.nn.l2_normalize(x,0)

theta=weight_variable([34452,1])
bias=tf.random_normal([1])


result=tf.matmul(xx_normal,theta)+bias

loss=tf.reduce_mean(tf.square(result-y))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)


saver=tf.train.Saver()


#init = tf.initialize_all_variables()


with tf.Session() as sess:
    #sess.run(init)
    saver.restore(sess,"save.ckpt")
    for i in range(1000):
        if i%10==0:
            print sess.run(loss,feed_dict={x:house_info,y:price_info})
            print "save in :"+saver.save(sess,"save.ckpt")
        sess.run(train_step,feed_dict={x:house_info,y:price_info})



