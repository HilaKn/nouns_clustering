class ParserOutputWrapper(object):

    TOKEN_ID_COLUMN = 0
    TOKEN_COLUMN = 1
    POS_COLUMN = 3
    DEP_ID_COLUMN = 6
    DEP_RELATION_COLUMN = 7

    NO_HEAD_NOUN_TAG = '<no_head_noun_tag>'
    ADJ_TAG = 'JJ'
    NOUN_TAGS = ['NN','NNS']#proper nouns'NNP','NNPS']
    NON_RELEVANT_HEAD = ['.',',']