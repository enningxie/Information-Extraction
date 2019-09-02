# coding=utf-8
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam


class Model03():
    def __init__(self, config):
        self.config = config

    def seq_gather(self, x):
        """seq是[None, seq_len, s_size]的格式，
        idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
        最终输出[None, s_size]的向量。
        """
        seq, idxs = x
        idxs = K.cast(idxs, 'int32')
        batch_idxs = K.arange(0, K.shape(seq)[0])
        batch_idxs = K.expand_dims(batch_idxs, 1)
        idxs = K.concatenate([batch_idxs, idxs], 1)
        return K.tf.gather_nd(seq, idxs)

    def get_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config.config_path, self.config.checkpoint_path,
                                                        seq_len=None)
        # whether trainable
        for l in bert_model.layers:
            l.trainable = True

        # Input
        # segment embedding
        text_in_1 = Input(shape=(None,))
        # position embedding
        text_in_2 = Input(shape=(None,))
        # aspect start index
        a_s_in = Input(shape=(1,))
        a_s_in_ = Input(shape=(None,))
        # aspect end index
        a_e_in = Input(shape=(1,))
        a_e_in_ = Input(shape=(None,))
        # opinion start index
        o_s_in = Input(shape=(1,))
        o_s_in_ = Input(shape=(None,))
        # opinion end index
        o_e_in = Input(shape=(1,))
        o_e_in_ = Input(shape=(None,))
        # category
        c_in = Input(shape=(1,))
        # polarity
        p_in = Input(shape=(1,))

        mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(text_in_1)

        t = bert_model([text_in_1, text_in_2])
        # ps1 ps2
        a_s_out = Dense(1, activation='sigmoid')(t)
        a_e_out = Dense(1, activation='sigmoid')(t)

        #
        o_s_out = Dense(1, activation='sigmoid')(t)
        o_e_out = Dense(1, activation='sigmoid')(t)

        aspect_model = Model([text_in_1, text_in_2], [a_s_out, a_e_out])  # 预测aspect的模型
        opinion_model = Model([text_in_1, text_in_2], [o_s_out, o_e_out])  # 预测opinion的模型

        # aspect part
        asv = Lambda(self.seq_gather)([t, a_s_in])
        aev = Lambda(self.seq_gather)([t, a_e_in])
        av = Average()([asv, aev])
        a = Add()([t, av])
        a_ = Bidirectional(LSTM(100))(a)
        category_out = Dense(13, activation='softmax')(a_)
        category_model = Model([text_in_1, text_in_2, a_s_in, a_e_in], [category_out])

        # opinion part
        osv = Lambda(self.seq_gather)([t, o_s_in])
        oev = Lambda(self.seq_gather)([t, o_e_in])
        ov = Average()([osv, oev])
        o = Add()([t, ov])
        o_ = Bidirectional(LSTM(100))(o)
        polarity_out = Dense(3, activation='softmax')(o_)
        polarity_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [polarity_out])

        train_model = Model([text_in_1, text_in_2, a_s_in, a_e_in, o_s_in, o_e_in],
                            [a_s_out, a_e_out, o_s_out, o_e_out, category_out, polarity_out])

        # aspect loss
        a_s_in_ = K.expand_dims(a_s_in_, 2)
        a_e_in_ = K.expand_dims(a_e_in_, 2)

        a_s_loss = K.binary_crossentropy(a_s_in_, a_s_out)
        a_s_loss = K.sum(a_s_loss * mask) / K.sum(mask)
        a_e_loss = K.binary_crossentropy(a_e_in_, a_e_out)
        a_e_loss = K.sum(a_e_loss * mask) / K.sum(mask)

        # opinion loss
        o_s_in_ = K.expand_dims(o_s_in_, 2)
        o_e_in_ = K.expand_dims(o_e_in_, 2)

        o_s_loss = K.binary_crossentropy(o_s_in_, o_s_out)
        o_s_loss = K.sum(o_s_loss * mask) / K.sum(mask)
        o_e_loss = K.binary_crossentropy(o_e_in_, o_e_out)
        o_e_loss = K.sum(o_e_loss * mask) / K.sum(mask)

        # category loss
        c_loss = K.sparse_categorical_crossentropy(c_in, category_out)
        c_loss = K.sum(c_loss * mask) / K.sum(mask)

        # polarity loss
        p_loss = K.sparse_categorical_crossentropy(p_in, polarity_out)
        p_loss = K.sum(p_loss * mask) / K.sum(mask)

        total_loss = (a_s_loss + a_e_loss) + (o_s_loss + o_e_loss) + c_loss + p_loss

        train_model.add_loss(total_loss)
        train_model.compile(optimizer=Adam(self.config.learning_rate))
        train_model.summary()
        plot_model(train_model, 'model/model_03.png', show_shapes=True)
        return train_model, aspect_model, opinion_model, category_model, polarity_model
