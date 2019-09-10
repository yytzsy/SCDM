import tensorflow as tf
import math
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from utils import *
from opt import *

options = default_options()

def atrous_conv2d(x,input_filters,output_filters,kernel,rate,name):
    with tf.variable_scope(name) as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.constant(0.0, shape=[output_filters])
        bias = tf.get_variable('bias', initializer=bias)
        x_conv = tf.nn.atrous_conv2d(x, weight, rate=rate, padding='SAME')
        return tf.nn.bias_add(x_conv,bias)



def multi_dilate(x,input_filters,is_training,name):
    with tf.variable_scope(name) as scope:
        conv1=tf.nn.relu(conv2d(x,input_filters,32,3,1,name='conv1'))
        conv2_rate1=conv2d(conv1,32,32,3,1,name='conv2')
        conv2_rate2=atrous_conv2d(conv1,32,32,3,rate=2,name='astrous1')
        conv2_rate4 = atrous_conv2d(conv1, 32, 32,3, rate=4, name='astrous2')
        conv2_rate6 = atrous_conv2d(conv1, 32, 32,3, rate=6, name='astrous3')
        conv2_concat = tf.concat([conv2_rate1,conv2_rate2,conv2_rate4,conv2_rate6],axis=3)
        conv2_concat = conv2d(conv2_concat,128,32,1,1,name='conv3')
        conv2_BN = batch_norm_layer(conv2_concat,is_training)
        return tf.nn.relu(conv2_BN)


def residual(x,channels=64,kernel_size=3,name='residual'):
    with tf.variable_scope(name) as scope:
        tmp = conv2d(x,channels,channels,kernel_size,1,name='conv1')
        tmp = tf.nn.relu(tmp)
        tmp = conv2d(tmp,channels,channels,kernel_size,1,name='conv2')
        # tmp = Squeeze_excitation_layer(tmp, channels, 16,'squeeze_excitation')
    return x + tmp*0.1

def conv2d(x, input_filters, output_filters, kernel_h,kernel_w, strides, name='conv'):
    with tf.variable_scope(name) as scope:
        shape = [kernel_h, kernel_w, input_filters, output_filters]
        weight = tf.get_variable('weight',shape=shape,initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.constant(0.0, shape=[output_filters])
        bias = tf.get_variable('bias', initializer=bias)
        x_conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name='conv')
        return tf.nn.bias_add(x_conv,bias)


def batch_norm_layer(x, train_phase):
    bn_train = batch_norm(x, decay=0.99, center=True, scale=True, updates_collections=None, is_training=train_phase,
                          reuse=None, trainable=True, scope='bn')
    return bn_train


def bn_relu_conv(image, is_training, input_filters, output_filters, kernel_h,kernel_w, strides,name='bn_relu_conv1'):
    with tf.variable_scope(name) as scope:
        image_bn = batch_norm_layer(image, train_phase=is_training)
        image_relu = tf.nn.relu(image_bn)
        image_conv = conv2d(image_relu, input_filters, output_filters, kernel_h,kernel_w, strides)
    return image_conv

def bn_relu_upsample(image, is_training, input_filters, output_filters, kernel_h,kernel_w, strides,name='bn_relu_conv1'):
    with tf.variable_scope(name) as scope:
        image_bn = batch_norm_layer(image, train_phase=is_training)
        image_relu = tf.nn.relu(image_bn)
        image_upsample = upsample(image_relu, input_filters, output_filters, kernel_h,kernel_w, strides)
    return image_upsample

def add_layer(name, l, is_training, input_filters1, input_filters2=64, output_filters1=64, output_filters2=16,
              kernel1=1, kernel2=3, strides=1):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        c = bn_relu_conv(l, is_training, input_filters1, output_filters1, kernel1, strides,name='bn_relu_conv1')
        c = bn_relu_conv(c, is_training, input_filters2, output_filters2, kernel2, strides,name='bn_relu_conv2')
        l = tf.concat([c, l], 3)
    return l


def add_transition_average(name, l, is_training, input_filters, output_filters):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = bn_relu_conv(l, is_training, input_filters, output_filters, 1, 1)
        l_pool = tf.nn.avg_pool(l,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    return l, l_pool


def upsample(l, input_filters, output_filters, kernel_h,kernel_w, strides):
    shape = l.get_shape().as_list()
    shape[2] = shape[2] * strides
    shape[3] = output_filters
    weight_shape = [kernel_h, kernel_w, output_filters, input_filters]
    weight = tf.get_variable('weight', shape=weight_shape, initializer=tf.contrib.layers.xavier_initializer())
    upsample_result = tf.nn.conv2d_transpose(l, weight,
                                             output_shape=shape,
                                             strides=[1, 1, strides, 1], padding="SAME")
    return upsample_result

def sub_pixel_shuffle(x, r,batch,height,width):
    batch = batch
    H = height
    W = width
    C = (int)(288 // (r*r))
    t = tf.reshape(x, [batch, H, W, r, r, C])
    t = tf.transpose(t, perm=[0, 1, 3, 2, 4, 5])  # S, H, r, H, r, C
    t = tf.reshape(t, [batch, H*r, W*r, C])
    return t




def bidirectional_GRU(options, inputs, inputs_len, cell = None, cell_fn = tf.contrib.rnn.GRUCell, units = 256, layers = 1, scope = "Bidirectional_GRU", output = 0, is_training = True, reuse = None):
    '''
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        cell:       rnn cell of type RNN_Cell.
        output:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    '''
    with tf.variable_scope(scope, reuse = reuse):
        if cell is not None:
            (cell_fw, cell_bw) = cell
        else:
            shapes = inputs.get_shape().as_list()
            if len(shapes) > 3:
                inputs = tf.reshape(inputs,(shapes[0]*shapes[1],shapes[2],-1))
                inputs_len = tf.reshape(inputs_len,(shapes[0]*shapes[1],))

            # if no cells are provided, use standard GRU cell implementation
            if layers > 1:
                cell_fw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
                cell_bw = MultiRNNCell([apply_dropout(cell_fn(units), size = inputs.shape[-1] if i == 0 else units, is_training = is_training) for i in range(layers)])
            else:
                cell_fw, cell_bw = [apply_dropout(options, cell_fn(units), size = inputs.shape[-1], is_training = is_training) for _ in range(2)]
                
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        sequence_length = inputs_len,
                                                        dtype=tf.float32)
        if output == 0:
            return tf.concat(outputs, 2)
        elif output == 1:
            return tf.reshape(tf.concat(states,1),(options['batch_size'], shapes[1], 2*units))



def avg_sentence_pooling(options, memory, units, memory_len = None, scope = "avg_sentence_pooling"):
    with tf.variable_scope(scope):
        avg_attn = tf.cast(tf.cast(tf.sequence_mask(memory_len, maxlen=options['max_sen_len']),tf.int32),tf.float32)
        avg_attn = tf.div(avg_attn, tf.cast(tf.tile(tf.expand_dims(memory_len,1),[1,options['max_sen_len']]),tf.float32) )
        attn = tf.expand_dims(avg_attn, -1)
        return tf.reduce_sum(attn * memory, 1) #, avg_attn


def gated_attention(options, memory, inputs, states, units, params, self_matching = False, memory_len = None, scope="gated_attention"):
    with tf.variable_scope(scope):
        weights, W_g = params
        inputs_ = [memory, inputs]
        states = tf.reshape(states,(options['batch_size'],options['sentence_hidden_dim']))
        if not self_matching:
            inputs_.append(states)
        scores = attention(inputs_, units, weights, memory_len = memory_len)
        save_attention = scores
        scores = tf.expand_dims(scores,-1)
        attention_pool = tf.reduce_sum(scores * memory, 1)
        inputs = tf.concat((inputs,attention_pool),axis = 1)
        g_t = tf.sigmoid(tf.matmul(inputs,W_g))
        return g_t * inputs


def apply_dropout(options, inputs, size = None, is_training = True):
    '''
    Implementation of Zoneout from https://arxiv.org/pdf/1606.01305.pdf
    '''
    if options['dropout'] is None and options['zoneout'] is None:
        return inputs
    if options['zoneout'] is not None:
        return ZoneoutWrapper(inputs, state_zoneout_prob= options['zoneout'], is_training = is_training)
    elif is_training:
        return tf.contrib.rnn.DropoutWrapper(inputs,
                                            output_keep_prob = 1 - options['dropout'],
                                            # variational_recurrent = True,
                                            # input_size = size,
                                            dtype = tf.float32)
    else:
        return inputs


def mask_attn_score(score, memory_sequence_length, score_mask_value = -1e8):
    score_mask = tf.sequence_mask(memory_sequence_length, maxlen=score.shape[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(score_mask, score, score_mask_values)


def attention(inputs, units, weights, scope = "attention", memory_len = None, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        outputs_ = []
        weights, v = weights
        for i, (inp,w) in enumerate(zip(inputs,weights)):
            shapes = inp.shape.as_list()
            inp = tf.reshape(inp, (-1, shapes[-1]))
            if w is None:
                w = tf.get_variable("w_%d"%i, dtype = tf.float32, shape = [shapes[-1],options['dim_hidden']], initializer = tf.contrib.layers.xavier_initializer())
            outputs = tf.matmul(inp, w)
            # Hardcoded attention output reshaping. Equation (4), (8), (9) and (11) in the original paper.
            if len(shapes) > 2:
                outputs = tf.reshape(outputs, (shapes[0], shapes[1], -1))
            elif len(shapes) == 2 and shapes[0] is options['batch_size']:
                outputs = tf.reshape(outputs, (shapes[0],1,-1))
            else:
                outputs = tf.reshape(outputs, (1, shapes[0],-1))
            outputs_.append(outputs)
        outputs = sum(outputs_)
        if options['bias']:
            b = tf.get_variable("b", shape = outputs.shape[-1], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            outputs += b
        scores = tf.reduce_sum(tf.tanh(outputs) * v, [-1])
        if memory_len is not None:
            scores = mask_attn_score(scores, memory_len)
        return tf.nn.softmax(scores) # all attention output is softmaxed now