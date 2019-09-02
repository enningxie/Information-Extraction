import pickle
from collections import defaultdict
import numpy as np
from model.config import Config
from random import choice

# tmp_config = Config()


def gen_total_data_dict():
    # 恢复数据
    with open('../data/total_data.pickle', 'rb') as file:
        total_data = pickle.load(file)

    print(total_data[0])
    print(len(total_data))

    total_data_dict = defaultdict(list)

    for tmp_data in total_data:
        total_data_dict[tmp_data[0]].append(tmp_data[1:])

    print(len(total_data_dict))
    counter = 0
    for tmp_key in total_data_dict:
        print('----------')
        print(tmp_key)
        print(total_data_dict[tmp_key])
        counter += len(total_data_dict[tmp_key])
    print(counter)

    # # 保存数据
    # 感觉还可以，唯一的遗憾就是面膜放了香精，希望***官方能把香精去掉就完美了 t
    # [[(17, 19), (0, 0), 1, 2], [(0, 0), (2, 5), 11, 0]] a o c p
    # with open('../data/total_data_dict.pickle', 'wb') as f:
    #     pickle.dump(total_data_dict, f, -1)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def process_data_01(data, add_fake=False, a_or_o=(1, 0)):
    # text_in_1, text_in_2, a_s_in, a_e_in, o_s_in, o_e_in, a_s_in_, a_e_in_, o_s_in_, o_e_in_, c_in, p_in
    t1 = []
    t2 = []
    a_s_in_, a_e_in_ = [], []
    a_s_in, a_e_in = [], []
    co1_in, co2_in = [], []
    po1_in, po2_in = [], []
    counter1_ = 0
    counter2_ = 0

    for tmp_key in data:
        t1_, t2_ = tmp_config.tokenizer.encode(first=tmp_key)
        tmp_opinion_list = []
        for tmp_data_ in data[tmp_key]:
            tmp_opinion_list.append(tmp_data_[a_or_o[0]])
        for tmp_data in data[tmp_key]:
            t1.append(t1_)
            t2.append(t2_)

            tmp_a_s_in_, tmp_a_e_in_ = np.zeros(len(t1_)), np.zeros(len(t1_))
            tmp_aspect = tmp_data[a_or_o[0]]
            if tmp_aspect[0] == tmp_aspect[1]:
                tmp_a_s_in_[0] = 1
                tmp_a_e_in_[0] = 1
                tmp_a_s_in = 0
                tmp_a_e_in = 0
            else:
                tmp_a_s_in_[tmp_aspect[0] + 1] = 1
                tmp_a_e_in_[tmp_aspect[1]] = 1
                tmp_a_s_in = tmp_aspect[0] + 1
                tmp_a_e_in = tmp_aspect[1]
            a_s_in_.append(tmp_a_s_in_)
            a_e_in_.append(tmp_a_e_in_)
            a_s_in.append([tmp_a_s_in])
            a_e_in.append([tmp_a_e_in])

            tmp_co1_in, tmp_co2_in = np.zeros((len(t1_), 13)), np.zeros((len(t1_), 13))
            tmp_opinion = tmp_data[a_or_o[1]]
            tmp_category = tmp_data[2]
            if tmp_opinion[0] == tmp_opinion[1]:
                tmp_co1_in[0][tmp_category] = 1
                tmp_co2_in[0][tmp_category] = 1
            else:
                tmp_co1_in[tmp_opinion[0] + 1][tmp_category] = 1
                tmp_co2_in[tmp_opinion[1]][tmp_category] = 1
            co1_in.append(tmp_co1_in)
            co2_in.append(tmp_co2_in)

            tmp_po1_in, tmp_po2_in = np.zeros((len(t1_), 3)), np.zeros((len(t1_), 3))
            tmp_opinion = tmp_data[a_or_o[1]]
            tmp_polarity = tmp_data[3]
            if tmp_opinion[0] == tmp_opinion[1]:
                tmp_po1_in[0][tmp_polarity] = 1
                tmp_po2_in[0][tmp_polarity] = 1
            else:
                tmp_po1_in[tmp_opinion[0] + 1][tmp_polarity] = 1
                tmp_po2_in[tmp_opinion[1]][tmp_polarity] = 1
            po1_in.append(tmp_po1_in)
            po2_in.append(tmp_po2_in)

            if add_fake and len(tmp_opinion_list) > 1:
                o1, o2 = np.asarray(tmp_opinion_list).T
                counter1 = 0
                for tmp_o1 in o1:
                    for tmp_o2 in o2[o2 > tmp_o1]:
                        if (tmp_o1, tmp_o2) not in tmp_opinion_list:
                            a_s_in.append([tmp_o1])
                            a_e_in.append([tmp_o2])
                            t1.append(t1_)
                            t2.append(t2_)
                            a_s_in_.append(tmp_a_s_in_)
                            a_e_in_.append(tmp_a_e_in_)
                            co1_in.append(np.zeros((len(t1_), 13)))
                            co2_in.append(np.zeros((len(t1_), 13)))
                            po1_in.append(np.zeros((len(t1_), 3)))
                            po2_in.append(np.zeros((len(t1_), 3)))
                            counter1 += 1
                            if counter1 >= len(tmp_opinion_list):
                                break
                    if counter1 >= len(tmp_opinion_list):
                        break
                counter1_ += counter1


                # counter2 = 0
                # for tmp_o2 in o2:
                #     for tmp_o1 in o1[o1 < tmp_o2]:
                #         if (tmp_o1, tmp_o2) not in tmp_opinion_list:
                #             a_s_in.append([tmp_o1])
                #             a_e_in.append([tmp_o2])
                #             t1.append(t1_)
                #             t2.append(t2_)
                #             a_s_in_.append(tmp_a_s_in_)
                #             a_e_in_.append(tmp_a_e_in_)
                #             co1_in.append(np.zeros((len(t1_), 13)))
                #             co2_in.append(np.zeros((len(t1_), 13)))
                #             po1_in.append(np.zeros((len(t1_), 3)))
                #             po2_in.append(np.zeros((len(t1_), 3)))
                #             counter2 += 1
                # counter2_ += counter2


            # if add_fake:
            #     # old one
            #     for i in range(2):
            #         t1.append(t1_)
            #         t2.append(t2_)
            #         a_s_in_.append(tmp_a_s_in_)
            #         a_e_in_.append(tmp_a_e_in_)
            #         a_s_in.append([tmp_a_s_in])
            #         flag = tmp_a_e_in
            #         while flag == tmp_a_e_in:
            #             flag = choice(list(range(tmp_a_s_in + 1, len(t1_))))
            #         a_e_in.append([flag])
            #         co1_in.append(np.zeros((len(t1_), 13)))
            #         co2_in.append(np.zeros((len(t1_), 13)))
            #         po1_in.append(np.zeros((len(t1_), 3)))
            #         po2_in.append(np.zeros((len(t1_), 3)))
            #
            #     if tmp_a_e_in != 0:
            #         flag = tmp_a_s_in
            #         while flag == tmp_a_s_in:
            #             flag = choice(list(range(tmp_a_e_in)))
            #         a_s_in.append([flag])
            #         a_e_in.append([tmp_a_e_in])
            #         co1_in.append(np.zeros((len(t1_), 13)))
            #         co2_in.append(np.zeros((len(t1_), 13)))
            #         po1_in.append(np.zeros((len(t1_), 3)))
            #         po2_in.append(np.zeros((len(t1_), 3)))
            #         t1.append(t1_)
            #         t2.append(t2_)
            #         a_s_in_.append(tmp_a_s_in_)
            #         a_e_in_.append(tmp_a_e_in_)
    print('s->e: {}'.format(counter1_))
    print('s<-e: {}'.format(counter2_))
    text_in_1 = seq_padding(t1)
    text_in_2 = seq_padding(t2)
    a_s_in_ = seq_padding(a_s_in_)
    a_e_in_ = seq_padding(a_e_in_)
    a_s_in, a_e_in = np.asarray(a_s_in), np.asarray(a_e_in)
    co1_in = seq_padding(co1_in, np.zeros(13))
    co2_in = seq_padding(co2_in, np.zeros(13))
    po1_in = seq_padding(po1_in, np.zeros(3))
    po2_in = seq_padding(po2_in, np.zeros(3))

    return text_in_1, text_in_2, a_s_in, a_e_in, a_s_in_, a_e_in_, co1_in, co2_in, po1_in, po2_in

def get_dev_data():
    with open('data/total_data_dict.pickle', 'rb') as file:
        total_data = pickle.load(file)
        print(len(total_data))

    dev_data_len = len(total_data) // 10
    dev_text = []
    counter = 0
    total_data_ = {}
    for tmp_key in total_data:
        if counter < dev_data_len:
            dev_text.append(tmp_key)
            counter += 1
        else:
            total_data_[tmp_key] = total_data[tmp_key]

    return total_data_, dev_text

def process_data_02(data, add_fake=False):
    # text_in_1, text_in_2, a_s_in, a_e_in, o_s_in, o_e_in, a_s_in_, a_e_in_, o_s_in_, o_e_in_, c_in, p_in
    t1 = []
    t2 = []
    a_s_in_, a_e_in_ = [], []
    a_s_in, a_e_in = [], []
    o_s_in_, o_e_in_ = [], []
    o_s_in, o_e_in = [], []
    c_in, p_in = [], []

    for tmp_key in data:
        t1_, t2_ = tmp_config.tokenizer.encode(first=tmp_key)
        opinion_list = []
        #
        for tmp_data_ in data[tmp_key]:
            opinion_list.append(tmp_data_[1])
        # a o c p
        for tmp_data in data[tmp_key]:
            # positive
            t1.append(t1_)
            t2.append(t2_)

            tmp_a_s_in_, tmp_a_e_in_ = np.zeros(len(t1_)), np.zeros(len(t1_))
            tmp_aspect = tmp_data[0]
            if tmp_aspect[0] == tmp_aspect[1]:
                tmp_a_s_in_[0] = 1
                tmp_a_e_in_[0] = 1
                tmp_a_s_in = 0
                tmp_a_e_in = 0
            else:
                tmp_a_s_in_[tmp_aspect[0] + 1] = 1
                tmp_a_e_in_[tmp_aspect[1]] = 1
                tmp_a_s_in = tmp_aspect[0] + 1
                tmp_a_e_in = tmp_aspect[1]
            a_s_in_.append(tmp_a_s_in_)
            a_e_in_.append(tmp_a_e_in_)
            a_s_in.append([tmp_a_s_in])
            a_e_in.append([tmp_a_e_in])

            tmp_o_s_in_, tmp_o_e_in_ = np.zeros(len(t1_)), np.zeros(len(t1_))
            tmp_opinion = tmp_data[1]
            if tmp_opinion[0] == tmp_opinion[1]:
                tmp_o_s_in_[0] = 1
                tmp_o_e_in_[0] = 1
                tmp_o_s_in = 0
                tmp_o_e_in = 0
            else:
                tmp_o_s_in_[tmp_opinion[0] + 1] = 1
                tmp_o_e_in_[tmp_opinion[1]] = 1
                tmp_o_s_in = tmp_opinion[0] + 1
                tmp_o_e_in = tmp_opinion[1]
            o_s_in_.append(tmp_o_s_in_)
            o_e_in_.append(tmp_o_e_in_)
            o_s_in.append([tmp_o_s_in])
            o_e_in.append([tmp_o_e_in])

            tmp_c_in, tmp_p_in = np.zeros(13), np.zeros(3)
            tmp_c, tmp_p = tmp_data[2], tmp_data[3]
            tmp_c_in[tmp_c] = 1
            tmp_p_in[tmp_p] = 1
            c_in.append(tmp_c_in)
            p_in.append(tmp_p_in)

            if add_fake:
                # fake data
                # 1
                t1.append(t1_)
                t2.append(t2_)
                a_s_in_.append(tmp_a_s_in_)
                a_e_in_.append(tmp_a_e_in_)
                a_s_in.append([tmp_a_s_in])
                flag = tmp_a_e_in
                while flag == tmp_a_e_in:
                    flag = choice(list(range(tmp_a_s_in + 1, len(t1_))))
                a_e_in.append([flag])
                o_s_in_.append(tmp_o_s_in_)
                o_e_in_.append(tmp_o_e_in_)
                o_s_in.append([tmp_o_s_in])
                o_e_in.append([tmp_o_e_in])
                c_in.append(np.zeros(13))
                p_in.append(np.zeros(3))

                # 2
                t1.append(t1_)
                t2.append(t2_)
                a_s_in_.append(tmp_a_s_in_)
                a_e_in_.append(tmp_a_e_in_)
                a_s_in.append([tmp_a_s_in])
                a_e_in.append([tmp_a_e_in])
                o_s_in_.append(tmp_o_s_in_)
                o_e_in_.append(tmp_o_e_in_)
                o_s_in.append([tmp_o_s_in])
                flag = tmp_o_e_in
                while flag == tmp_o_e_in:
                    flag = choice(list(range(tmp_o_s_in + 1, len(t1_))))
                o_e_in.append([flag])
                c_in.append(np.zeros(13))
                p_in.append(np.zeros(3))

                # 3
                t1.append(t1_)
                t2.append(t2_)
                a_s_in_.append(tmp_a_s_in_)
                a_e_in_.append(tmp_a_e_in_)
                a_s_in.append([tmp_a_s_in])
                flag = tmp_a_e_in
                while flag == tmp_a_e_in:
                    flag = choice(list(range(tmp_a_s_in + 1, len(t1_))))
                a_e_in.append([flag])
                o_s_in_.append(tmp_o_s_in_)
                o_e_in_.append(tmp_o_e_in_)
                o_s_in.append([tmp_o_s_in])
                flag = tmp_o_e_in
                while flag == tmp_o_e_in:
                    flag = choice(list(range(tmp_o_s_in + 1, len(t1_))))
                o_e_in.append([flag])
                c_in.append(np.zeros(13))
                p_in.append(np.zeros(3))

            # if tmp_a_e_in != 0:
            #     flag = tmp_a_s_in
            #     while flag == tmp_a_s_in:
            #         flag = choice(list(range(tmp_a_e_in)))
            #     a_s_in.append([flag])
            #     a_e_in.append([tmp_a_e_in])
            #     co1_in.append(np.zeros((len(segment_embedding), 13)))
            #     co2_in.append(np.zeros((len(segment_embedding), 13)))
            #     po1_in.append(np.zeros((len(segment_embedding), 3)))
            #     po2_in.append(np.zeros((len(segment_embedding), 3)))
            #     segment_embeddings.append(segment_embedding)
            #     position_embeddings.append(position_embedding)
            #     a_s_in_.append(tmp_a_s_in_)
            #     a_e_in_.append(tmp_a_e_in_)

    text_in_1 = seq_padding(t1)
    text_in_2 = seq_padding(t2)
    a_s_in_ = seq_padding(a_s_in_)
    a_e_in_ = seq_padding(a_e_in_)
    a_s_in, a_e_in = np.asarray(a_s_in), np.asarray(a_e_in)
    o_s_in_ = seq_padding(o_s_in_)
    o_e_in_ = seq_padding(o_e_in_)
    o_s_in, o_e_in = np.asarray(o_s_in), np.asarray(o_e_in)
    c_in = seq_padding(c_in)
    p_in = seq_padding(p_in)

    # o_s_in_ = pad_sequences(o_s_in_, maxlen=90, padding='post', truncating='post')
    # o_e_in_ = pad_sequences(o_e_in_, maxlen=90, padding='post', truncating='post')
    # o_s_in, o_e_in = np.asarray(o_s_in), np.asarray(o_e_in)

    return text_in_1, text_in_2, a_s_in, a_e_in, o_s_in, o_e_in, a_s_in_, a_e_in_, o_s_in_, o_e_in_, c_in, p_in


if __name__ == '__main__':
    with open('../opinion.pkl', 'rb') as file:
        total_data = pickle.load(file)

    for tmp_data in total_data:
        print(tmp_data)






