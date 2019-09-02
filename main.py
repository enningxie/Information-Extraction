# coding=utf-8
import os
from model.config import Config
from model.model_01 import Model01
from model.model_02 import Model02
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.data_utils import process_data_01, process_data_02, get_dev_data
import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from keras.callbacks import Callback
import keras.backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# with fake a -> o
model_path = 'saved_models/model_01_xz_01.weights'


class Runner():
    def __init__(self):
        self.config = Config()
        # model_01
        self.train_model, self.a_or_o_model, self.category_model, self.polarity_model = Model01(self.config).get_model()
        # # model_02
        # self.train_model, self.aspect_model, self.opinion_model, self.category_model, self.polarity_model = Model02(
        #     self.config).get_model()

    def train_op(self):
        total_data, dev_text = get_dev_data()
        train_labels = pd.read_csv('data/Train_labels.csv')
        train_reviews = pd.read_csv('data/Train_reviews.csv')

        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0,
                                     save_best_only=True, save_weights_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto', baseline=None,
                                   restore_best_weights=True)
        train_model_ = self.train_model
        config = self.config
        a_or_o_model = self.a_or_o_model
        category_model = self.category_model
        polarity_model = self.polarity_model

        def extract_items(text_in):
            # aspect_model, object_model
            _tokens = config.tokenizer.tokenize(text_in)
            _t1, _t2 = config.tokenizer.encode(first=text_in)
            _t1, _t2 = np.array([_t1]), np.array([_t2])
            _a1_, _a2_ = a_or_o_model.predict([_t1, _t2])
            _a1, _a2 = np.where(_a1_[0] > 0.1)[0], np.where(_a2_[0] > 0.1)[0]
            _aspects = []
            for i in _a1:
                j = _a2[_a2 >= i]
                if len(j) > 0:
                    j = j[0]
                    if i == j:
                        _aspect = '_'
                        _aspects.append((_aspect, 0, 0))
                    else:
                        _aspect = text_in[i - 1: j]
                        _aspects.append((_aspect, i, j))
            # category_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [ca1, ca2])
            # polarity_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [pa1, pa2])
            if _aspects:
                R = []
                # error_text = []
                _t1 = np.repeat(_t1, len(_aspects), 0)
                _t2 = np.repeat(_t2, len(_aspects), 0)
                _a1, _a2 = np.array([_s[1:] for _s in _aspects]).T.reshape((2, -1, 1))
                co1_out, co2_out = category_model.predict([_t1, _t2, _a1, _a2])
                po1_out, po2_out = polarity_model.predict([_t1, _t2, _a1, _a2])
                for i, _aspect in enumerate(_aspects):
                    c1_prob = np.max(co1_out[i])
                    c2_prob = np.max(co2_out[i])
                    if c1_prob > c2_prob:
                        os1, c1 = np.where(co1_out[i] == c1_prob)
                        oe1, c2 = np.where(co2_out[i] == np.max(co2_out[i][os1[0]:, c1[0]]))
                    else:
                        oe1, c2 = np.where(co2_out[i] == c2_prob)
                        os1, c1 = np.where(co1_out[i] == np.max(co1_out[i][:oe1[0] + 1, c2[0]]))
                    ###
                    # bug len(os1) > 1
                    tmp_os = os1[0]
                    tmp_oe = oe1[0]
                    tmp_c = c2[0]
                    p1_prob = np.max(po1_out[i][tmp_os])
                    p2_prob = np.max(po2_out[i][tmp_oe])
                    if p1_prob > p2_prob:
                        tmp_p = np.argmax(po1_out[i][tmp_os]).item()
                    else:
                        tmp_p = np.argmax(po2_out[i][tmp_oe]).item()
                    tmp_data = []

                    if tmp_os == tmp_oe:
                        tmp_data.append('_')
                    else:
                        tmp_data.append(text_in[tmp_os - 1: tmp_oe])
                    tmp_data.append(_aspect[0])
                    tmp_data.append(config.categories[tmp_c])
                    tmp_data.append(config.polarities[tmp_p])
                    R.append(tmp_data)
                    ##
                return R
            else:
                return []

        class Evaluate(Callback):
            def __init__(self):
                self.F1 = []
                self.best = 0.
                self.passed = 0
                self.stage = 0
                self.learning_rate = 5e-5
                self.min_learning_rate = 1e-5

            # def on_batch_begin(self, batch, logs=None):
            #     """第一个epoch用来warmup，第二个epoch把学习率降到最低
            #     """
            #     if self.passed < self.params['steps']:
            #         lr = (self.passed + 1.) / self.params['steps'] * self.learning_rate
            #         K.set_value(self.model.optimizer.lr, lr)
            #         self.passed += 1
            #     elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            #         lr = (2 - (self.passed + 1.) / self.params['steps']) * (self.learning_rate - self.min_learning_rate)
            #         lr += self.min_learning_rate
            #         K.set_value(self.model.optimizer.lr, lr)
            #         self.passed += 1

            def on_epoch_end(self, epoch, logs=None):
                f1, precision, recall = self.evaluate()
                self.F1.append(f1)
                if f1 > self.best:
                    self.best = f1
                    train_model_.save_weights(model_path)
                print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

            def evaluate(self):
                A, B, C = 1e-10, 1e-10, 1e-10
                for tmp_text in tqdm(dev_text):
                    tmp_id = train_reviews[train_reviews.Reviews == tmp_text].id.values[0]
                    R = extract_items(tmp_text)
                    R_ = set()
                    if not R:
                        R_.add(('_', '_', '_', '_'))
                    else:
                        for tmp_r in R:
                            R_.add((tmp_r[0], tmp_r[1], tmp_r[2], tmp_r[3]))
                    T_ = set()
                    for tmp_t in train_labels[train_labels.id == tmp_id].values:
                        T_.add((tmp_t[1], tmp_t[4], tmp_t[7], tmp_t[8]))

                    A += len(R_ & T_)
                    B += len(R_)
                    C += len(T_)
                return 2 * A / (B + C), A / B, A / C

        train_model_.fit(
            x=[*process_data_01(total_data, add_fake=True, a_or_o=(1, 0))],
            batch_size=32,
            epochs=30,
            validation_split=0.1,
            callbacks=[Evaluate()]
        )

    def extract_items_01(self, text_in):
        # aspect_model, object_model
        _tokens = self.config.tokenizer.tokenize(text_in)
        _t1, _t2 = self.config.tokenizer.encode(first=text_in)
        _t1, _t2 = np.array([_t1]), np.array([_t2])
        _a1_, _a2_ = self.a_or_o_model.predict([_t1, _t2])
        _a1, _a2 = np.where(_a1_[0] > 0.1)[0], np.where(_a2_[0] > 0.1)[0]
        _aspects = []
        for i in _a1:
            j = _a2[_a2 >= i]
            if len(j) > 0:
                j = j[0]
                if i == j:
                    _aspect = '_'
                    _aspects.append((_aspect, 0, 0))
                else:
                    _aspect = text_in[i - 1: j]
                    _aspects.append((_aspect, i, j))
        # category_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [ca1, ca2])
        # polarity_model = Model([text_in_1, text_in_2, o_s_in, o_e_in], [pa1, pa2])
        if _aspects:
            R = []
            # error_text = []
            _t1 = np.repeat(_t1, len(_aspects), 0)
            _t2 = np.repeat(_t2, len(_aspects), 0)
            _a1, _a2 = np.array([_s[1:] for _s in _aspects]).T.reshape((2, -1, 1))
            co1_out, co2_out = self.category_model.predict([_t1, _t2, _a1, _a2])
            po1_out, po2_out = self.polarity_model.predict([_t1, _t2, _a1, _a2])
            for i, _aspect in enumerate(_aspects):
                c1_prob = np.max(co1_out[i])
                c2_prob = np.max(co2_out[i])
                if c1_prob > c2_prob:
                    os1, c1 = np.where(co1_out[i] == c1_prob)
                    oe1, c2 = np.where(co2_out[i] == np.max(co2_out[i][os1[0]:, c1[0]]))
                else:
                    oe1, c2 = np.where(co2_out[i] == c2_prob)
                    os1, c1 = np.where(co1_out[i] == np.max(co1_out[i][:oe1[0] + 1, c2[0]]))
                ###
                # os1, c1 = np.where(co1_out[i] == np.max(co1_out[i]))
                # oe1, c2 = np.where(co2_out[i] == np.max(co2_out[i][os1[0]:, c1[0]]))
                # os2, p1 = np.where(po1_out[i] == np.max(po1_out[i]))
                # oe2, p2 = np.where(po2_out[i] == np.max(po2_out[i]))
                # bug len(os1) > 1
                tmp_os = os1[0]
                tmp_oe = oe1[0]
                tmp_c = c2[0]
                p1_prob = np.max(po1_out[i][tmp_os])
                p2_prob = np.max(po2_out[i][tmp_oe])
                if p1_prob > p2_prob:
                    tmp_p = np.argmax(po1_out[i][tmp_os]).item()
                else:
                    tmp_p = np.argmax(po2_out[i][tmp_oe]).item()

                # _po1, _po2 = np.where(po1_out[i] > 0.15), np.where(po2_out[i] > 0.15)
                # candidate_p = []
                # for tmp_o1, tmp_p in zip(*_po1):
                #     if tmp_os == tmp_o1:
                #         candidate_p.append(tmp_p)
                # if not candidate_p:
                #     continue
                # tmp_max = 0
                # true_p = candidate_p[0]
                # for tmp_p_ in candidate_p:
                #     if po1_out[i][tmp_os][tmp_p_] > tmp_max:
                #         tmp_max = po1_out[i][tmp_os][tmp_p_]
                #         true_p = tmp_p_
                tmp_data = []
                if tmp_os == tmp_oe:
                    tmp_data.append('_')
                else:
                    tmp_data.append(text_in[tmp_os - 1: tmp_oe])
                tmp_data.append(_aspect[0])
                tmp_data.append(self.config.categories[tmp_c])
                tmp_data.append(self.config.polarities[tmp_p])
                R.append(tmp_data)
                ###
                # _oo1, _oo2 = np.where(co1_out[i] > 0.2), np.where(co2_out[i] > 0.2)
                # for _ooo1, _c1 in zip(*_oo1):
                #     for _ooo2, _c2 in zip(*_oo2):
                #         if _ooo1 <= _ooo2 and _c1 == _c2:
                #             _predicate = self.config.categories[_c1]
                #             if _ooo1 == _ooo2:
                #                 tmp_opinion = '_'
                #             else:
                #                 tmp_opinion = text_in[_ooo1 - 1: _ooo2]
                #             if not C[(_aspect[0], tmp_opinion)]:
                #                 C[(_aspect[0], tmp_opinion)].append(_predicate)
                #             break
                # _oo1_, _oo2_ = np.where(po1_out[i] > 0.2), np.where(po2_out[i] > 0.2)
                # for _ooo1, _c1 in zip(*_oo1_):
                #     for _ooo2, _c2 in zip(*_oo2_):
                #         if _ooo1 <= _ooo2 and _c1 == _c2:
                #             _predicate = self.config.polarities[_c1]
                #             if _ooo1 == _ooo2:
                #                 tmp_opinion = '_'
                #             else:
                #                 tmp_opinion = text_in[_ooo1 - 1: _ooo2]
                #             if not O[(_aspect[0], tmp_opinion)]:
                #                 O[(_aspect[0], tmp_opinion)].append(_predicate)
                #             break
            return R
        else:
            return []

    # def extract_items_02(self, text_in):
    #     # aspect_model, object_model
    #     _tokens = self.config.tokenizer.tokenize(text_in)
    #     _t1, _t2 = self.config.tokenizer.encode(first=text_in)
    #     _t1, _t2 = np.array([_t1]), np.array([_t2])
    #     _a1_, _a2_ = self.aspect_model.predict([_t1, _t2])
    #     _a1, _a2 = np.where(_a1_[0] > 0.2)[0], np.where(_a2_[0] > 0.2)[0]
    #     _o1_, _o2_ = self.opinion_model.predict([_t1, _t2])
    #     _o1, _o2 = np.where(_o1_[0] > 0.2)[0], np.where(_o2_[0] > 0.2)[0]
    #     _aspects = []
    #     _opinions = []
    #     for i in _a1:
    #         j = _a2[_a2 >= i]
    #         if len(j) > 0:
    #             j = j[0]
    #             if i == j:
    #                 _aspect = '_'
    #                 _aspects.append((_aspect, 0, 0))
    #             else:
    #                 _aspect = text_in[i - 1: j]
    #                 _aspects.append((_aspect, i, j))
    #     # for i in _o1:
    #     #     j = _o2[_o2 >= i]
    #     #     if len(j) > 0:
    #     #         j = j[0]
    #     #         if i == 0 and j == 0:
    #     #             _opinion = '-'
    #     #             _opinions.append((_opinion, 0, 0))
    #     #         else:
    #     #             _opinion = text_in[i - 1: j]
    #     #             _opinions.append((_opinion, i, j))
    #     # assert len(_aspects) == len(_opinions), '_aspects and _opinions length error.'
    #     if _aspects:
    #         C = defaultdict(list)
    #         O = defaultdict(list)
    #         _t1 = np.repeat(_t1, len(_aspects), 0)
    #         _t2 = np.repeat(_t2, len(_aspects), 0)
    #         _a1, _a2 = np.array([_s[1:] for _s in _aspects]).T.reshape((2, -1, 1))
    #         # _o1, _o2 = np.array([_s[1:] for _s in _opinions]).T.reshape((2, -1, 1))
    #         co1_out, co2_out, po1_out, po2_out = object_model.predict([_t1, _t2, _a1, _a2])
    #         for i, _subject in enumerate(_aspects):
    #             _oo1, _oo2 = np.where(co1_out[i] > 0.1), np.where(co2_out[i] > 0.1)
    #             for _ooo1, _c1 in zip(*_oo1):
    #                 for _ooo2, _c2 in zip(*_oo2):
    #                     if _ooo1 <= _ooo2 and _c1 == _c2:
    #                         _predicate = categories[_c1]
    #                         if _ooo1 == _ooo2:
    #                             tmp_opinion = '_'
    #                         else:
    #                             tmp_opinion = text_in[_ooo1 - 1: _ooo2]
    #                         if not C[(_subject[0], tmp_opinion)]:
    #                             C[(_subject[0], tmp_opinion)].append(_predicate)
    #                         break
    #             _oo1, _oo2 = np.where(po1_out[i] > 0.05), np.where(po2_out[i] > 0.05)
    #             for _ooo1, _c1 in zip(*_oo1):
    #                 for _ooo2, _c2 in zip(*_oo2):
    #                     if _ooo1 <= _ooo2 and _c1 == _c2:
    #                         _predicate = polarities[_c1]
    #                         if _ooo1 == _ooo2:
    #                             tmp_opinion = '_'
    #                         else:
    #                             tmp_opinion = text_in[_ooo1 - 1: _ooo2]
    #                         if not O[(_subject[0], tmp_opinion)]:
    #                             O[(_subject[0], tmp_opinion)].append(_predicate)
    #                         break
    #         # _c_out = np.argmax(c_out, axis=-1)
    #         # _p_out = np.argmax(p_out, axis=-1)
    #         # for i in range(len(_aspects)):
    #         #     R.append((_aspects[i][0], _opinions[i][0], categories[_c_out[i]], polarities[_p_out[i]]))
    #         return C, O
    #     else:
    #         return [], []

    def get_result(self, csv_path):
        train_reviews = pd.read_csv(csv_path)
        ids_ = []
        aspects_ = []
        opinions_ = []
        categories_ = []
        polarities_ = []
        error_texts = []
        for tmp_value in tqdm(train_reviews.values):
            tmp_id = tmp_value[0]
            tmp_text = tmp_value[1]
            R = self.extract_items_01(tmp_text)
            # error_texts.extend(error_text)
            if R:
                for tmp_r in R:
                    ids_.append(tmp_id)
                    aspects_.append(tmp_r[0])
                    opinions_.append(tmp_r[1])
                    categories_.append(tmp_r[2])
                    polarities_.append(tmp_r[3])
            else:
                ids_.append(tmp_id)
                aspects_.append('_')
                opinions_.append('_')
                categories_.append('_')
                polarities_.append('_')

        result_ = pd.DataFrame(
            {'id': ids_, 'aspect': aspects_, 'opinion': opinions_, 'category': categories_, 'polarity': polarities_})
        result_.drop_duplicates(inplace=True)
        result_.to_csv('data/result_xz_01.csv', index=False)
        print('Export Result.csv success.')
        # for tmp_error_text in set(error_texts):
        #     print(tmp_error_text)

    def run(self, train_flag=True, debug_flag=False, evaluate_flag=False):
        if train_flag:
            self.train_op()
        self.train_model.load_weights(model_path)
        if debug_flag:
            R = self.extract_items_01('上妆效果：容易上妆 个人情况：油性皮肤 持久情况：下午好像有点出油 整体描述：还不错 适合肤质：油性肤质')
            print(R)
        if evaluate_flag:
            self.get_result('data/Train_reviews.csv')


if __name__ == '__main__':
    runner = Runner()
    runner.run(train_flag=False, debug_flag=True, evaluate_flag=False)
