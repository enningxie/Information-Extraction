# coding=utf-8
from keras_bert import Tokenizer
import codecs


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class Config():
    def __init__(self):
        self.config_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
        self.checkpoint_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        self.dict_path = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

        self.categories = ['包装', '成分', '尺寸', '服务', '功效', '价格', '气味', '使用体验', '物流', '新鲜度', '真伪', '整体', '其他']
        self.polarities = ['正面', '中性', '负面']

        self.tokenizer = self.gen_tokenizer()

        # self.save_model_weights_path = 'saved_models/model_02.weights'
        self.total_data_dict_path = 'data/total_data_dict.pickle'

        self.learning_rate = 5e-5

        self.maxlen = 90

    def gen_tokenizer(self):
        token_dict = {}

        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return OurTokenizer(token_dict)
