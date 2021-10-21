# models.py

import numpy as np
import collections
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import math

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier):
    def predict(self, context):
        raise Exception("Implement me")


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


class TextDataset(Dataset):
    def __init__(self, cons_exs, vowel_exs, vocab_index):
        self.examples = []
        for sent in cons_exs:
            c_array = [char for char in sent]
            c_idx = []
            for c in c_array:
                c_idx.append(vocab_index.index_of(c))
            self.examples.append([c_idx, 0])
        for sent in vowel_exs:
            c_array = [char for char in sent]
            c_idx = []
            for c in c_array:
                c_idx.append(vocab_index.index_of(c))
            self.examples.append([c_idx, 0])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = torch.tensor(self.examples[idx][0])
        label = torch.tensor(self.examples[idx][1])
        return text, label


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_size, output_size):
        super(LSTMTagger, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(vocab_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(vocab_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #embeds = self.word_embeddings(input)
        #input = torch.squeeze(input)
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """

    train_data = TextDataset(train_cons_exs, train_vowel_exs, vocab_index)
    test_data = TextDataset(dev_cons_exs, dev_vowel_exs, vocab_index)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 20
    TAG_SET_SIZE = 2

    rnn = LSTMTagger(len(vocab_index), HIDDEN_DIM, EMBEDDING_DIM, TAG_SET_SIZE)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn

    def train(category_tensor, line_tensor):
        hidden = rnn.initHidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()

    epochs = 100
    print_every = 5
    plot_every = 1

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(epochs):
        for line_tensor, category_tensor in train_dataloader:

            output, loss = train(category_tensor, line_tensor)
            current_loss += loss

        print('%d %d%% ' % (timeSince(start), loss, ))


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self):
        raise Exception("Implement me")

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")
