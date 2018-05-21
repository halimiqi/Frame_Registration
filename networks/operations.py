#@Time     :2018/4/20 13:45
import tensorflow as tf

def conv2d(input, outputSize, kernelSize, stride, initializer = tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernelShape = [kernelSize[0], kernelSize[1], input.get_shape()[1], outputSize]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernelShape = [kernelSize[0], kernelSize[1], input.get_shape()[-1], outputSize]

        w = tf.get_variable('w', kernelShape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(input, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [outputSize], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

    if activation_fn != None:
        out = activation_fn(out)

    return out, w, b

def conv3d(input,
           outputDim,
           kernelSize,  #the kernel format is Depth, Height, Width
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NDHWC',
           padding='VALID',
           name='conv3d'):
  with tf.variable_scope(name):
    if data_format == 'NDHWC':
      stride = [1,stride[0], stride[1], stride[2],1]
      kernelShape = [kernelSize[0], kernelSize[1], kernelSize[2] , input.get_shape()[-1], outputDim]
    elif data_format == 'NCDHW':
      stride = [1,1, stride[0], stride[1], stride[2]]
      kernelShape = [kernelSize[0], kernelSize[1],kernelSize[2],  input.get_shape()[1], outputDim]

    w = tf.get_variable('w', kernelShape, tf.float32, initializer=initializer)
    conv = tf.nn.conv3d(input, w, stride, padding, data_format=data_format)

    b = tf.get_variable('biases', [outputDim], initializer=tf.constant_initializer(0.0))
    if data_format == 'NDHWC':
        out = tf.nn.bias_add(conv, b)
    elif data_format == 'NCDHW':
        tempconv = tf.transpose(conv, perm=[0, 2, 3,4,1])
        out = tf.add(tempconv,b)
        out = tf.transpose(out, perm = [0,4,1,2,3])

    #out = tf.add(conv, b)
  if activation_fn != None:
    out = activation_fn(out)  #add the activation function of the w*x+b

  return out, w, b


def linear(input, outputSize, stddev=0.02, biasStart=0.0, activation_fn=None, name='linear'):  #stddev is standard deviation this layer is full connected layer
  shape = input.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], outputSize], tf.float32,
        tf.contrib.layers.xavier_initializer())   #创建新的变量
    b = tf.get_variable('bias', [outputSize],
        initializer=tf.constant_initializer(biasStart))

    out = tf.nn.bias_add(tf.matmul(input, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b

def avg_pooling(input, kernelShape, strides, data_format, padding='SAME', name='avgpooling' ):
    if data_format == "NHWC":
        kernelShape = [1,kernelShape[0], kernelShape[1], 1]
        strides = [1,strides[0],strides[1], 1]
    else: # the data format is NCHW
        kernelShape = [1, 1, kernelShape[0], kernelShape[1]]
        strides = [1,1,strides[0], strides[1]]
    out = tf.nn.avg_pool(input,ksize=kernelShape, strides=strides, padding=padding,data_format=data_format, name = name)
    return out

def max_pooling(input ,kernelShape, strides, data_format, padding='SAME', name = 'maxpooling'):
    if data_format == "NHWC":
        kernelShape = [1,kernelShape[0], kernelShape[1], 1]
        strides = [1,strides[0],strides[1], 1]
    else: # the data format is NCHW
        kernelShape = [1, 1, kernelShape[0], kernelShape[1]]
        strides = [1,1,strides[0], strides[1]]
    out = tf.nn.max_pool(input,ksize=kernelShape, strides=strides, padding=padding,data_format=data_format, name = name)
    return out

def IfOverfitting(loss, minloss):
    if loss>= 1.5* minloss:
        return True , minloss
    elif loss>=minloss:
        return False , minloss
    elif loss< minloss:
        return False, loss
