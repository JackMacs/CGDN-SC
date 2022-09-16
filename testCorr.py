import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec
import argparse
import math
#注意yto的数据sentence的长度为30
# sentences = ["仓租费是什么", "仓族费是什么", "仓纽费是什么", "仓妲费是什么", "仓做费是什么", "仓迥费是什么", "仓作费是什么"]
model = Word2Vec.load('word_vector/wordVec_model/word2vecModel_pub')
def wordToVector(words):
    # print('words:',words)
    result = []
    for senarr in words:
        temp = []
        for i in range(30):
            if i < len(senarr):
                try:
                    word_vec = model[senarr[i]]
                    word_vec = np.asarray(word_vec)
                except:
                    print('word2vec no word:', senarr[i])
                    # word_vec = np.random.random(128)
                    word_vec = [1 / 128 for c in range(128)]
                    word_vec = np.asarray(word_vec)
                    print('word_vec_shape:', word_vec.shape)
                    print('word_vec:', word_vec)
            else:
                word_vec = [0 for _ in range(128)]
            temp.append(word_vec)
        # temp = np.asarray(temp)
        result.append(temp)
    return result

def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial, name='w')

def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial, name='b')


def lstm_cell(num_units, keep_prob=1):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

def getSen():
    # sens = open('/correctionData/candidates_yt.txt', encoding='utf-8').read().split('\n')
    sens = open('/Users/fighting/PycharmProjects/Text_error_detection/correctionData/candidates1.txt', encoding='utf-8').read().split('\n')
    r = []
    for str in sens:
        sarr = str.split(" ")
        r.append(sarr)
    return r


def main():
    # sentencess = [["亲怎么上门收见", "亲怎么上门收捡", "亲怎么上门收接", "亲怎么上门收现", "亲怎么上门收加", "亲怎么上门收江", "亲怎么上门收件"], ["仓租费是什么", "仓族费是什么", "仓纽费是什么", "仓妲费是什么", "仓做费是什么", "仓迥费是什么", "仓作费是什么"]]
    h = 0
    r = getSen()
    sentences = r[5]
    print("测试的sentences的长度:")
    print(len(sentences))
    h += 1
    tests = []
    for sen in sentences:
        tests.append(list(sen))
    test_x = wordToVector(tests)
    test_x = np.asarray(test_x)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)
    print("test_x.shape[0]:", test_x.shape[0], "test_steps:", test_steps, "FLAGS.test_batch_size",
          FLAGS.test_batch_size)
    # global_step = tf.Variable(-1, trainable=False, name='global_step')
    global_step = tf.Variable(-1, trainable=True, name='global_step')
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x))
    test_dataset = test_dataset.batch(FLAGS.train_batch_size)
    iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
    with tf.name_scope('test_dataset_initial'):
        test_initializer = iterator.make_initializer(test_dataset)
    with tf.name_scope('inputs'):
        x = iterator.get_next()
    x = tf.cast(x, dtype=tf.float32)
    inputs = x
    print('inputs Reshape', inputs)
    # Variables
    keep_prob = tf.placeholder(tf.float32, [])
    is_train = tf.placeholder(tf.bool)
    # st = tf.placeholder(tf.int32, [])
    with tf.name_scope('biLSTM_Cell_Layer'):
        # RNN Layer
        # cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
        # cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)])
        with tf.name_scope("LSTM_Cell_fw"):
            cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
        with tf.name_scope("LSTM_Cell_bw"):
            cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
        # initial_state_fw = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
        # initial_state_bw = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
        h_lstm = tf.unstack(inputs, FLAGS.time_step, axis=1)
        output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=h_lstm, dtype=tf.float32)
        h_bilstm = tf.stack(output, axis=1)
        print('h_bilstm', h_bilstm)
        h_bilstm = tf.reshape(h_bilstm, [-1, FLAGS.num_units * 2])
        h_bilstm = tf.tanh(h_bilstm)
        # output = tf.layers.batch_normalization(output, training= is_train)
        print('h_bilstm Reshape', h_bilstm)

    with tf.name_scope('outputs'):
        w4 = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b4 = bias([FLAGS.category_num])
        y = tf.matmul(h_bilstm, w4) + b4
        y = tf.reshape(y, [-1, 30, FLAGS.category_num])
        y = tf.cast(y, tf.float64)
        y_o = tf.reduce_sum(y, 1)
        y_o = tf.layers.batch_normalization(y_o, training=is_train)
        # y = tf.sigmoid(y)  # 将输出值进行归一化
        # y_o = tf.reshape(y_o, [])
        y_ = tf.nn.softmax(y_o)
        print("Y:", y)
        y_predict = tf.cast(tf.argmax(y_, axis=1), tf.int32)  # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
        print('Output Y', y_predict)
    # 预测
    # Saver
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(test_initializer)
    ckpt = tf.train.get_checkpoint_state('ckpt4')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restore from', ckpt.model_checkpoint_path)
    for step in range(int(test_steps)):
        y_predict_result, y_r, yf = sess.run([y_predict, y_,y], feed_dict={keep_prob: FLAGS.keep_prob, is_train:True})
        ypr = np.asarray(y_predict_result)
        print(ypr)
        yf = np.asarray(yf)
        print(yf.shape)
        y_r = np.asarray(y_r)
        i = 0
        max = -1
        flag = -1
        print(sentences)
        for t in y_r:
            if ypr[i] != 1:
                i += 1
                # continue
            else:
                if max < t[ypr[i]]:
                    flag = i
                    max = t[ypr[i]]
                print(t)
                print(t[ypr[i]])
                print(sentences[i])
                i += 1

        print("纠正的结果为："+sentences[flag])
        print("概率为:", max)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--train_batch_size', help='train batch size', default=1000)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=50)
    parser.add_argument('--test_batch_size', help='test batch size', default=800)  # 模型每次测试的样本
    parser.add_argument('--source_data', help='source size', default='./data/data_word3vec.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)
    parser.add_argument('--num_units', help='num of units', default=50, type=int)  #num_units输出向量的维度
    # parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--time_step', help='time steps', default=30, type=int)   # sentence的长度
    parser.add_argument('--embedding_size', help='time steps', default=128, type=int)
    # parser.add_argument('--category_num', help='category num', default=5, type=int)
    parser.add_argument('--category_num', help='category num', default=2, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)
    # parser.add_argument('--epoch_num', help='num of epoch', default=1000, type=int)
    parser.add_argument('--epoch_num', help='num of epoch', default=50, type=int)
    parser.add_argument('--epochs_per_test', help='epochs per test', default=10, type=int)
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=100, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=1, type=float)
    # parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.6, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt4/model_corr.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    parser.add_argument('--train', help='train', default=False, type=bool)
    # parser.add_argument('--train', help='train', default=True, type=bool)
    FLAGS, args = parser.parse_known_args()
    main()
