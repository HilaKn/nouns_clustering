import numpy as np
import os
import gensim
import gzip
from parser_output_wrapper import ParserOutputWrapper as parser_w
from collections import defaultdict
from commons import *
import operator
import random
from nltk.stem import WordNetLemmatizer

check_list = set(["after","afterward","thereafter","later","afterwards","afterwords","before","following"])

class WordsData(object):

    def __init__(self,words_list,matrix):
        self.words = words_list
        self.matrix = matrix

    @property
    def dataset_size(self):
        return len(self.words)

class DataProcessor(object):

    def __init__(self,we_file, input_file,is_noun_freq_file= False): #load the trained word vectors
        self.we_file = we_file
        self.input_file = input_file
        self.is_noun_freq_file_flag = is_noun_freq_file

    def extract_nouns_from_parser_output(self, nouns):
        with gzip.open(self.input_file, 'rb') as f:
            sentence_id = 1
            sentence_data = []
            for line in f:
                if line != '\n':
                    data = line.split('\t')
                    sentence_data.append(data)


                else:
                    for word_data in sentence_data:
                        if word_data[parser_w.POS_COLUMN] == parser_w.ADJ_TAG:
                            noun_candidate_id = int(word_data[parser_w.DEP_ID_COLUMN])
                            candidate_data = sentence_data[noun_candidate_id-1]
                            word = candidate_data[parser_w.TOKEN_COLUMN]
                            pos = candidate_data[parser_w.POS_COLUMN]
                            if pos in parser_w.NOUN_TAGS:
                                nouns[word] += 1

                                if word in check_list:
                                    print "{} is a noun ???!!!!".format(candidate_data[parser_w.TOKEN_COLUMN])
                    if (sentence_id % 1000000 == 0):
                        print "finished process sentence {}".format(sentence_id)
                        # break
                    sentence_id+=1
                    sentence_data = []

        nouns_file = os.path.join(INTERNAL_FOLDER, NOUNS_OUT_FILE)
        sorted_nouns = sorted(nouns.items(), key=operator.itemgetter(1), reverse=True)
        with open(nouns_file, 'a') as fo:
            for noun_data in sorted_nouns:
                row = "{}\t{}\n".format(noun_data[0], str(noun_data[1]))
                fo.write(row)

    #found some issues with this method so implemented another one. The problem is there are POS mistakes in the input
    #like "becasue", "if" and other words tagged as noun. I'll fix it by taking only the nouns that adjective is related to them
    # def extract_nouns_from_parser_output(self, nouns):
    #     with gzip.open(self.input_file, 'rb') as f:
    #         sentence_id = 1
    #         sentence_data = []
    #         for line in f:
    #             if line != '\n':
    #                 data = line.split('\t')
    #                 sentence_data.append(data)
    #                 word = data[parser_w.TOKEN_COLUMN].lower()
    #                 pos = data[parser_w.POS_COLUMN]
    #                 if pos in parser_w.NOUN_TAGS:
    #                     nouns[word] += 1
    #                     if word in ["whenever","therefore", "if", "though", "because"]:
    #                         print "{} is a noun ???!!!!".format(word)
    #
    #             else:
    #                 if (sentence_id % 1000000 == 0):
    #                     print "finished process sentence {}".format(sentence_id)
    #                     # break
    #                 sentence_id+=1
    #                 sentence_data = []
    #
    #     nouns_file = os.path.join(INTERNAL_FOLDER, "nouns")
    #     sorted_nouns = sorted(nouns.items(), key=operator.itemgetter(1), reverse=True)
    #     with open(nouns_file, 'a') as fo:
    #         for noun_data in sorted_nouns:
    #             row = "{}\t{}\n".format(noun_data[0], str(noun_data[1]))
    #             fo.write(row)

    def get_nuons_with_freq(self):
        print "start extracting nouns"
        nouns = defaultdict(int)
        if self.is_noun_freq_file_flag:
            with open(self.input_file, 'r') as f:
                for row in f:
                    row_data = row.rstrip('\n').split('\t')
                    nouns[row_data[0]] = int(row_data[1])

        else:

            self.extract_nouns_from_parser_output(nouns)

        return nouns


    def prepare_data(self):
        print "Loading word vectors from {}".format(self.we_file)
        self.we_model = gensim.models.KeyedVectors.load(self.we_file, mmap='r') .wv # mmap the large matrix as read-only
        self.we_model.syn0norm = self.we_model.syn0
        lemmatizer = WordNetLemmatizer()
        
        nouns_freq = self.get_nuons_with_freq()
        filtered_nouns_list = [noun for noun,freq in nouns_freq.iteritems()
                                    if freq > MIN_NOUN_FREQ
                                    and noun in self.we_model.vocab]

        nouns_lemmas = list(set([lemmatizer.lemmatize(noun) for noun in filtered_nouns_list]))
        filtered_nouns_lemmas = [noun for noun in nouns_lemmas if noun in self.we_model.vocab]
        random.shuffle(filtered_nouns_lemmas)
        nouns_matrix =np.array([self.we_model.word_vec(noun) for noun in filtered_nouns_lemmas]).squeeze()
        nouns_data = WordsData(filtered_nouns_lemmas,nouns_matrix)
        
        return nouns_data
        