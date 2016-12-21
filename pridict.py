#coding:utf-8
import tensorflow as tf
import numpy as np
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def process(i,data):
    info=i.split(',')
    info[len(info)-1]=info[len(info)-1].replace('\n','')
    info=info[1:]

    for j in range(0,len(info)):
        info[j]=get_num(info[j],data[j])
    return info[len(info)-1],info[0:len(info)-1]

def process2(i,data):
    info=i.split(',')
    info[len(info)-1]=info[len(info)-1].replace('\n','')
    info=info[1:]

    for j in range(0,len(info)):
        info[j]=get_num(info[j],data[j])
    return info

def get_num(info_j,data_k):
    if data_k==True:
        if info_j=='NA':
            return 0.0
        return float(info_j)
    else:
        if data_k.has_key(info_j):
            return data_k[info_j]
        else:
            now=data_k["now"]
            data_k['now']=now+1
            data_k[info_j]=now
            return now



def get_data(i):
    info = i.split(',')
    info[len(info) - 1] = info[len(info) - 1].replace('\n', '')
    info_i = info[1:]
    data=list()
    for i in info_i:
        try:
            float(i)
            data.append(True)
        except:
            data.append({"now":5})
    return data
def get_cv(house_info):
    new_house=list()
    for house in house_info:
        now=list()
        for j in range(len(house)):
            for k in range(j,len(house)):
                now.append(house[j]*house[k])
        new_house.append(now)
    return new_house

file=open("/Users/zhangxu/Desktop/train.csv",'r')
lines=file.readlines()
lines=lines[1:]
data=get_data(lines[0])

house_info=list()
price_info=list()
for i in range(0,len(lines)):
    y,x=process(lines[i],data)
    house_info.append(x)
    price_info.append(y)




file=open("/Users/zhangxu/Desktop/test.csv",'r')
lines=file.readlines()
lines=lines[1:]


house_info=list()
price_info=list()
for i in range(0,len(lines)):
    x=process2(lines[i],data)
    house_info.append(x)


#print data




house_info=np.float32(house_info)

house_info_cv=get_cv(house_info)
house_info_new=np.hstack((house_info,house_info_cv))
print house_info_new.shape



#以上是对测试集特征进行相同的处理

#开始利用训练得到的参数(通过136行 saver.restore(sess,"save.ckpt") 从磁盘加载数据),  进行房价预测



x=tf.placeholder(dtype=tf.float32,shape=[None,3239])



xx_normal=tf.nn.l2_normalize(x,0)

theta=weight_variable([3239,1])
bias=tf.random_normal([1])


result=tf.matmul(xx_normal,theta)+bias



saver=tf.train.Saver()


#init = tf.initialize_all_variables()


with tf.Session() as sess:
    #sess.run(init)
    saver.restore(sess,"save.ckpt")
    res=sess.run(result,feed_dict={x:house_info_new})
    res=list(res)

    count=1461
    file=open('result.csv','w')
    for i in range(len(res)):
        line=str(count) + ',' + str(res[i][0] )+'\n'
        count=count+1
        file.write(line)
    file.close()



