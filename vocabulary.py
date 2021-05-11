
from global_hparams import voc_hparams

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = self.init_dict()
        self.num_words = 3  # Count SOS, EOS, PAD

    def init_dict(self):
        return {
            voc_hparams["PAD_token"]: "PAD",
            voc_hparams["SOS_token"]: "SOS",
            voc_hparams["EOS_token"]: "EOS"
            }
    def addSentence(self, sentence):
        for char in sentence:
            self.addWord(char)

    def addWord(self, char):
        if char not in self.word2index:
            self.word2index[char] = self.num_words
            self.word2count[char] = 1
            self.index2word[self.num_words] = char
            self.num_words += 1
        else:
            self.word2count[char] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = self.init_dict()
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)