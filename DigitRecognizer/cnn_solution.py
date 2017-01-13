import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd



'''加载数据'''
mnist = pd.read_csv(r'data/train.csv')
train_labels = mnist['label']
train_images = mnist.iloc[:,1:]
train_images.astype(np.float)
train_images = np.multiply(train_images, 1.0/255.0)
train_images = train_images.as_matrix()
train_labels = train_labels.as_matrix()


def compute_accuracy(xs,ys,X,y,keep_prob,sess,prediction):
    y_pre = sess.run(prediction,feed_dict={xs:X,keep_prob:1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:X,ys:y,keep_prob:1.0})
    return result  


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

def conv2d(x,W):#x是图片的所有参数，W是此卷积层的权重
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')#池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]




epochs_compeleted = 0
index_in_epoch = 0
def cnn():
    #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  
    xs = tf.placeholder(tf.float32,[None,784])
    ys = tf.placeholder(tf.float32,[None,10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs,[-1,28,28,1]) #-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
    
    # conv layer1
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # conv layer2
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
    predict = tf.argmax(prediction, 1)
    
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i in range(2000):
        batch_xs,batch_ys = next_batch(mnist, batch_size=100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

    mnist_test = pd.read_csv(r'data/test.csv')
    mnist_test.astype(np.float)
    mnist_test = np.multiply(mnist_test,1.0/255.0)
    X = mnist_test.as_matrix()
    BATCH_SIZE = 100
    predictions = np.zeros(mnist_test.shape[0])
    for i in range(mnist_test.shape[0]//BATCH_SIZE):   # 一批一批的预测，否则内存可能不够，这里4G
        predictions[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = sess.run(predict,feed_dict={xs:X[i*BATCH_SIZE : (i+1)*BATCH_SIZE],keep_prob:1.0})

    result = pd.DataFrame(data={'ImageId':range(1,X.shape[0]+1),'Label':predictions.astype(np.int32)})
    result.to_csv(r'my_prediction.csv',index=False)    
    #np.savetxt('submission_softmax.csv', 
               #np.c_[range(1,len(test_images)+1),predicted_lables], 
               #delimiter=',', 
               #header = 'ImageId,Label', 
               #comments = '', 
               #fmt='%d')    


'''数据的映射，例如1-->[0,1,0,0,0,0,0,0,0,0]'''
def dense_to_one_hot(label_dense,num_classes):
    num_labels = label_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + label_dense.ravel()] = 1  # flat展开
    return labels_one_hot    
    
'''使用SGD随机梯度下降，所以指定next batch的训练集'''
def next_batch(mnist,batch_size):
    num_examples = mnist.shape[0]
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_compeleted
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > num_examples:
        epochs_compeleted += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]   
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples        
    end = index_in_epoch
    train_labels_one_hot = dense_to_one_hot(train_labels[start:end], num_classes=10)
    return train_images[start:end], train_labels_one_hot
    
    
            
if __name__ == '__main__':
    cnn()