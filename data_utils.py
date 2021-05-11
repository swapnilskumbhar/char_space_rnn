from vocabulary import Voc

from global_hparams import voc_hparams

from io import open
import re
import unicodedata
import torch

import itertools

class prep_data:
    def __init__(self, max_len):
        self.max_len = max_len
    
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    # Read query/response pairs and return a voc object
    def readVocs(self, datafile, corpus_name):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8').read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[self.normalizeString(s.replace('...','.')) for s in l.split('\t')] for l in lines]
        final_pairs = []
        for pair in pairs:
            final_pairs.append([pair[0].replace(' ',''),pair[0]])
            final_pairs.append([pair[1].replace(' ',''),pair[1]])
        voc = Voc(corpus_name)
        return voc, final_pairs

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(self, p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0]) < self.max_len and len(p[1]) < self.max_len

    # Filter pairs using filterPair condition
    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(self, corpus, corpus_name, datafile, save_dir):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs(datafile, corpus_name)
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[1])
        return voc, pairs
    
    def indexesFromSentence(self, voc, sentence):
        return [voc.word2index[word] for word in sentence] + [voc_hparams['EOS_token']]

    def zeroPadding(self, l, fillvalue=voc_hparams['PAD_token']):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self, l, value=voc_hparams['PAD_token']):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == voc_hparams['PAD_token']:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l, voc):
        indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def batch2TrainData(self, voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch, voc)
        output, mask, max_target_len = self.outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len

