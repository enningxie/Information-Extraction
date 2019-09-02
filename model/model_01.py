# coding=utf-8
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam


class Model01():
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
        o_s_in = Input(shape=(1,))
        o_s_in_ = Input(shape=(None,))
        # aspect end index
        o_e_in = Input(shape=(1,))
        o_e_in_ = Input(shape=(None,))
        # category
        ca1_in = Input(shape=(None, 13))
        ca2_in = Input(shape=(None, 13))
        # polarity
        pa1_in = Input(shape=(None, 3))
        pa2_in = Input(shape=(None, 3))

        mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(text_in_1)

        t = bert_model([text_in_1, text_in_2])

        o_s_out = Dense(1, activation='sigmoid')(t)
        o_e_out = Dense(1, activation='sigmoid')(t)

        opinion_model = Model([text_in_1, text_in_2], [o_s_out, o_e_out])

        osv = Lambda(self.seq_gather)([t, o_s_in])
        oev = Lambda(self.seq_gather)([t, o_e_in])
        ov = Average()([osv, oev])
        # t_0 = Lambda(lambda x: x*0)(t)
        o = Add()([t, ov])
        # ov = Add()([t_0, ov])



        ca1 = Dense(13, activation='sigmoid')(o)
        ca2 = Dense(13, activation='sigmoid')(o)
        pa1 = Dense(3, activation='sigmoid')(o)
        pa2 = Dense(3, activation='sigmoid')(o)
        # ###
        # pa1 = Dense(3, activation='sigmoid')(ov)
        # pa2 = Dense(3, activation='sigmoid')(ov)
        # ###
        category_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [ca1, ca2])
        polarity_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [pa1, pa2])
        train_model = Model(
            [text_in_1, text_in_2, o_s_in, o_e_in, o_s_in_, o_e_in_, ca1_in, ca2_in, pa1_in, pa2_in],
            [o_s_out, o_e_out, ca1, ca2, pa1, pa2])

        # aspect loss
        o_s_in_ = K.expand_dims(o_s_in_, 2)
        o_e_in_ = K.expand_dims(o_e_in_, 2)

        o_s_loss = K.binary_crossentropy(o_s_in_, o_s_out)
        o_s_loss = K.sum(o_s_loss * mask) / K.sum(mask)
        o_e_loss = K.binary_crossentropy(o_e_in_, o_e_out)
        o_e_loss = K.sum(o_e_loss * mask) / K.sum(mask)

        # category loss
        ca1_loss = K.sum(K.binary_crossentropy(ca1_in, ca1), 2, keepdims=True)
        ca1_loss = K.sum(ca1_loss * mask) / K.sum(mask)
        ca2_loss = K.sum(K.binary_crossentropy(ca2_in, ca2), 2, keepdims=True)
        ca2_loss = K.sum(ca2_loss * mask) / K.sum(mask)

        # polarity loss
        pa1_loss = K.sum(K.binary_crossentropy(pa1_in, pa1), 2, keepdims=True)
        pa1_loss = K.sum(pa1_loss * mask) / K.sum(mask)
        pa2_loss = K.sum(K.binary_crossentropy(pa2_in, pa2), 2, keepdims=True)
        pa2_loss = K.sum(pa2_loss * mask) / K.sum(mask)

        total_loss = (o_s_loss + o_e_loss) + (ca1_loss + ca2_loss) + (pa1_loss + pa2_loss)

        train_model.add_loss(total_loss)
        train_model.compile(optimizer=Adam(self.config.learning_rate))
        train_model.summary()
        plot_model(train_model, 'model/model_01_.png', show_shapes=True)
        return train_model, opinion_model, category_model, polarity_model
