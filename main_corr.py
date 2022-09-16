import argparse
import tensorflow as tf
import math
import numpy as np
import pickle
import random
from gensim.models.word2vec import Word2Vec
# Read origin data13
# model = Word2Vec.load('word_vector/wordVec_model/word2vecModel_yt')
model = Word2Vec.load('word_vector/wordVec_model/word2vecModel_pub')
def wordToVector(words):
    # print('words:',words)
    result = []
    for senarr in words:
        temp = []
        # for i in range(30):
        for i in range(30):
            if i < len(senarr):
                try:
                    word_vec = model[senarr[i]]
                    word_vec = np.asarray(word_vec)
                except:
                    print('word2vec no word:', senarr[i])
                    # word_vec = np.random.random(128)
                    word_vec = [1/128 for c in range(128)]
                    word_vec = np.asarray(word_vec)
                    print('word_vec_shape:', word_vec.shape)
                    print('word_vec:', word_vec)
            else:
                word_vec = [0 for _ in range(128)]
            temp.append(word_vec)
        # temp = np.asarray(temp)
        result.append(temp)
    return result
FLAGS = None
# 将标签转化为one-hot编码
def transform_one_hot(labels):
    one_hot = np.eye(2)[labels]
    return one_hot

def load_data():
    """
    Load data13 from pickle
    :return: Arrays
    """
    with open(FLAGS.source_data, 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        return data_x, data_y


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial, name='w')

def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial, name='b')


def lstm_cell(num_units, keep_prob=1):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)



def test_xTofile(test_x):
    print(test_x)
    with open('./data13/testData', 'w') as w:
        for s in test_x:
            w.write(s)
            w.write('\n')

def main():
    # Load data13
    # train_x, train_y= load_data()
    # 将带有标签的样本数据转化为列表
    sentences, sen_test = [], []
    # 读取训练文本数据
    train_text = open('/Users/fighting/PycharmProjects/Text_error_detection/correctionData/train_data_sighan.txt',
                      encoding='utf-8').read().split('\n')
    # 将训练集中的数据打乱
    random.shuffle(train_text)
    # 将训练数据中的文本和标签分离
    trains = []
    labels = []
    for str in train_text:
        sarr = str.split('@')
        text = sarr[0]
        print(sarr)
        label = int(sarr[1])
        tarr = list(text)
        tarr = np.asarray(tarr)
        trains.append(tarr)
        labels.append(label)
    data_x = wordToVector(trains)
    data_y = labels
    data_x = np.asarray(data_x)
    print('data_x_shape', data_x.shape)
    # # data_y = transform_one_hot(data_y)
    data_y = np.asarray(data_y)
    print('data_y_shape', data_y.shape)


    train_x = data_x
    train_y = data_y
    # train_y = transform_one_hot(train_y)
    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
    print("train_x.shape[0]:", train_x.shape[0], "train_steps:", train_steps, "FLAGS.train_batch_size",
          FLAGS.train_batch_size)

    # global_step = tf.Variable(-1, trainable=False, name='global_step')
    global_step = tf.Variable(-1, trainable=True, name='global_step')

    # Train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # train_dataset = tf.data13.Dataset.from_tensor_slices(train_x)
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    # A reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    with tf.name_scope('train_dataset_initial'):
        train_initializer = iterator.make_initializer(train_dataset)

    # Input Layer
    # with tf.variable_scope('inputs'):
    with tf.name_scope('inputs'):
        x, y_label = iterator.get_next()


    # y_label_arr = transform_one_hot(y_label)

    # Embedding Layer
    # with tf.variable_scope('embedding'):
    #     embedding = tf.Variable(tf.random_normal([vocab_size, FLAGS.embedding_size]), dtype=tf.float32)
    # inputs = tf.nn.embedding_lookup(embedding, x)  # 查找张量中的序号为x的
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
    # with tf.name_scope('hidden1'):
    #     w1 = weight([FLAGS.num_units * 2, 128])
    #     b1 = bias([128])
    #     hidden1 = tf.matmul(h_bilstm, w1) + b1
    #     print('hidden1', hidden1)
    #     # hidden1 = tf.layers.batch_normalization(hidden1, training=is_train)
    # # Output Layer
    # # with tf.variable_scope('outputs'):
    # with tf.name_scope('hidden2'):
    #     w2 = weight([128, FLAGS.num_units])
    #     b2 = bias([FLAGS.num_units])
    #     hidden2 = tf.matmul(hidden1, w2) + b2
    with tf.name_scope('outputs'):
        w4 = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b4 = bias([FLAGS.category_num])
        y = tf.matmul(h_bilstm, w4) + b4
        y = tf.reshape(y, [-1, 30, FLAGS.category_num])
        y = tf.cast(y, tf.float64)
        y_o = tf.reduce_sum(y, 1)
        y_o = tf.layers.batch_normalization(y_o, training= is_train)
        # y = tf.sigmoid(y)  # 将输出值进行归一化
        y_ = tf.nn.softmax(y_o)
        print("Y:", y)
        y_predict = tf.cast(tf.argmax(y_, axis=1), tf.int32)  # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引
        print('Output Y', y_predict)
    # tf.summary.histogram('y_predict', y_predict)
    # # 改变正在训练的数据中标签的维度，使其成为一维列向量
    # y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    # # 改变正在训练句子长度的维度，使其成为一维向量
    # sentence_len_reshape = tf.cast(tf.reshape(sentence_len, [-1]), tf.int32)
    # # 将句子长度映射成为bool矩阵，true的个数为句中汉字的个数
    # loss_mask = tf.sequence_mask(tf.to_int32(sentence_len_reshape), tf.to_int32(FLAGS.time_step))
    # # 将bool矩阵转化为数值矩阵，目的是消除尾填充造成的损失误差
    # loss_mask = tf.cast(tf.reshape(loss_mask, [-1]), tf.float32)
    # print('loss_mask:', loss_mask)
    # # 不考虑尾填充的预测值
    # y_predict = tf.cast(y_predict, tf.float32) * loss_mask
    # # y_predict = tf.cast(y_predict, tf.float32)
    # yy_label_reshape = tf.cast(y_label_reshape, tf.float32) * loss_mask
    # # y_predict = tf.cast(y_predict, tf.float32)
    # # yy_label_reshape = tf.cast(y_label_reshape, tf.float32)
    # # 当消除尾填充误差时计算准确度
    # # correct_prediction = tf.cast(tf.equal(y_predict, yy_label_reshape), tf.float32)
    # correct_prediction = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_,labels=self.target)
    # with tf.name_scope('accuracy'):
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     # accuracy = correct_sum / mask_sum
    #     tf.summary.scalar('accuracy', accuracy)
    # ww = tf.constant(50, dtype=tf.float32)
    # ww = tf.constant(100, dtype=tf.float32)
    # lossW = tf.cast(lossW, tf.float64)
    # # yy_label = tf.cast(y_label, tf.float64)
    # # matmul支持两个类型为float64tensor数据进行点相乘
    # class_weight = tf.constant([0.1, 1, 0.1], shape=[1,3], dtype=tf.float64)
    # weighted_label = tf.transpose(tf.matmul(lossW, tf.transpose(class_weight)))
    # weighted_label = tf.reshape(weighted_label, [-1])
    # print('weighted_label:', weighted_label)
    # Loss
    with tf.name_scope('loss'):
        # loss_before = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
        #                                                                               logits=tf.cast(y, tf.float32))*loss_mask
        # loss_before = tf.cast(loss_before, tf.float64)
        # # multiply元素对应相乘需要类型保持一致
        # loss_after_w = tf.multiply(weighted_label, loss_before)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label,
                                                                logits=y_o)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
        #                                                              logits=tf.cast(y, tf.float32))
        loss = tf.reduce_mean(loss)


        # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
        #                                                                               logits=tf.cast(y, tf.float32))*loss_mask)*ww
        # print('loss_after_w:', loss_after_w)
        # idx = tf.where(loss_after_w > 0)
        # loss_after_w = tf.gather_nd(loss_after_w, idx)
        # cross_entropy = tf.reduce_mean(loss_after_w) * ww
        # tf.summary.scalar('loss', cross_entropy)
    # print("y_type,y_label.type", type(y.shape), type(y_label_reshape.shape))
    # print('Prediction', correct_prediction, 'Accuracy', accuracy)

    # Train
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

    # Summaries
    # 合并所有的summary
    summaries = tf.summary.merge_all()

    # Saver
    saver = tf.train.Saver()

    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Global step
    gstep = 0
    # writer = tf.summary.FileWriter(join(FLAGS.summaries_dir, 'train'),
    #                                sess.graph)

    writer = tf.summary.FileWriter(FLAGS.summaries_dir,
                                   sess.graph)


    if FLAGS.train:

        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

        for epoch in range(FLAGS.epoch_num):
            print('epoch:', epoch, "epoch_num:", FLAGS.epoch_num)
            tf.train.global_step(sess, global_step_tensor=global_step)

            # Train 获取训练的数据
            sess.run(train_initializer)
            # yy_label_shape, yy = sess.run([y_label_reshape, y_predict], feed_dict={keep_prob: FLAGS.keep_prob})
            # print("yy_label_shape:",yy_label_shape)
            # print("yy:", yy)
            for step in range(int(train_steps)):
                # output, y_o = sess.run([output, y_o], feed_dict={keep_prob: FLAGS.keep_prob, is_train:True})
                # output = np.asarray(output)
                # y_o = np.asarray(y_o)
                # print(" output:",  output.shape)
                # print(" y_o:", y_o.shape)
                loss_, gstep, _ = sess.run([loss, global_step, train],
                                                     feed_dict={keep_prob: FLAGS.keep_prob, is_train:True})
                # Print log
                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss_)

                # if loss <= 0.025:
                #     saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
                #     return

            # # 验证数据训练
            # if epoch % FLAGS.epochs_per_dev == 0:
            #     # Dev
            #     sess.run(dev_initializer)
            #     sess.run(de_initializer)
            #     sess.run(dev_len_initializer)
            #     sess.run(dev_lossW_initializer)
            #     for step in range(int(dev_steps)):
            #         if step % FLAGS.steps_per_print == 0:
            #             print('Dev Accuracy', sess.run(accuracy,  feed_dict={keep_prob: FLAGS.keep_prob, is_train:True}),
            #                   'Step', step)

            # Save model

            if epoch % FLAGS.epochs_per_save == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)


        # plot_learning_curves(accuracy)
    #
    # else:
        ckpt = tf.train.get_checkpoint_state('ckpt4')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)

    #     for step in range(int(test_steps)):
    #         x_results, y_predict_results, acc, y_label_results, hidden_ = sess.run([x, y_predict, accuracy, y_label_reshape, hidden],
    #                                                                       feed_dict={keep_prob:FLAGS.keep_prob, is_train:True})
    #         print('Test step', step, 'Accuracy', acc)
    #         y_predict_results = np.reshape(y_predict_results, (x_results.shape[0], x_results.shape[1]))
    #         y_label_results = np.reshape(y_label_results, (x_results.shape[0], x_results.shape[1]))
    #         f = 0
    #         TE = 0    # of testing sentences with errors
    #         SE = 0    # of sentences the evaluated system reported to have errors
    #         DC = 0    # of sentences with correctly detected results
    #         DE = 0    # of sentences with correctly detected errors
    #         FPE = 0   # of sentences with false positive errors
    #         ANE = 0   # of testing sentences without errors
    #         ALL = 0   # of all testing sentences
    #         AWE = 0   # of testing sentences with errors
    #         CLD = 0   # of sentences with correct location detection
    #         CEL = 0   # of sentences with correct error locations
    #
    #         for i in range(len(x_results)):
    #             # print('hidden:', hidden_[i])
    #             y_predict_result, y_label_result = list(filter(lambda x: x, y_predict_results[i])), list(filter(lambda x: x, y_label_results[i]))
    #             y_predict_text, y_label_text = ''.join(id2tag[y_predict_result].values), \
    #                                            ''.join(id2tag[y_label_result].values)
    #             index = step * len(x_results) +  i
    #             if y_predict_text == y_label_text:
    #                 f += 1
    #             print(texts[index])
    #             print(y_predict_text, "  ", y_label_text, ' f', f)
    #             ALL += 1
    #             if 'e' in y_predict_text:
    #                 SE += 1
    #             if 'e' in y_label_text:
    #                 TE += 1                 # of testing sentences with errors
    #             if ('e' not in y_label_text and 'e' not in y_predict_text) or ('e' in y_label_text and 'e' in y_predict_text):
    #                 DC += 1
    #             if 'e' in y_label_text and 'e' in y_predict_text:
    #                 DE += 1
    #             # if ('e' in y_label_text and 'e' not in y_predict_text) or ('e' in y_predict_text and 'e' not in y_label_text):
    #             if ('e' in y_predict_text and 'e' not in y_label_text):
    #                 FPE += 1
    #             if 'e' not in y_label_text:
    #                 ANE += 1
    #             if 'e' in y_label_text:
    #                 AWE += 1
    #             if y_predict_text == y_label_text:
    #                 CLD += 1
    #             if y_predict_text == y_label_text and 'e' in y_predict_text:
    #                 CEL += 1
    #         print('SE:', SE, 'TE:', TE, 'DE:', DE, 'DC:', DC, 'FPE:', FPE, 'ANE:', ANE,'ALL:', ALL,  'AWE:', AWE, 'CLD:', CLD, 'CEL:', CEL)
    #         FAR = FPE / ANE
    #         DA  =  DC / ALL
    #         DP = DE / SE
    #         DR = DE / TE
    #         DF1 = 2 * DP * DR / (DP + DR)
    #         ELA = CLD / ALL
    #         ELP = CEL / SE
    #         ELR = CEL / TE
    #         ELF1 = 2 * ELP * ELR / (ELP + ELR)
    #         print('FAR:', FAR, 'DA:', DA, 'DP:', DP, 'DR:', DR, 'DF1:', DF1, 'ELA:', ELA, 'ELP:', ELP, 'ELR:', ELR, 'ELF1:', ELF1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--train_batch_size', help='train batch size', default=500)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=50)
    parser.add_argument('--test_batch_size', help='test batch size', default=6409)  # 模型每次测试的样本
    parser.add_argument('--source_data', help='source size', default='./data/data_yt_corr.pkl')
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
    # parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=6, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=100, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=1, type=float)
    # parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.6, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt4/model_corr.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    # parser.add_argument('--train', help='train', default=False, type=bool)
    parser.add_argument('--train', help='train', default=True, type=bool)
    FLAGS, args = parser.parse_known_args()
    main()
