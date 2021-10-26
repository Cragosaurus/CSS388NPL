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

VOWELS = ['a', 'e', 'i', 'o', 'u', 'y']


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

    def __init__(self, rnn, vocab_indexer):
        self.rnn = rnn
        self.vocab_indexer = vocab_indexer

    def predict(self, context):
        self.rnn.eval()
        hidden = self.rnn.hidden_state(1)
        hidden = tuple([state.data for state in hidden])
        c_array = [char for char in context]
        c_idx = []
        for c in c_array:
            c_idx.append(self.vocab_indexer.index_of(c))
        x = c_idx[1:]
        x = torch.tensor(one_hot_encoder(x, len(self.vocab_indexer)))
        lstm_output, hidden = self.rnn.forward(x.view(1,19,len(self.vocab_indexer)), hidden)
        pred_idx = torch.topk(lstm_output,1)[1].data[-1].item()
        #print(f'Pred idx: {pred_idx}')
        pred_char = self.vocab_indexer.get_object(pred_idx)
        #print(f'Pred char: {pred_char}')
        if pred_char in VOWELS:
            #print('vowel')
            return 1
        else:
            return 0


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def one_hot_encoder(encoded_text, num_uni_char):
    encoded_text = np.asarray(encoded_text)
    one_hot = np.zeros((encoded_text.size, num_uni_char))
    one_hot = one_hot.astype(np.float32)
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_char))
    return one_hot


class TextDataset(Dataset):
    def __init__(self, cons_exs, vowel_exs, vocab_index):
        self.examples = []
        self.labels = []
        self.num_uni_char = len(vocab_index)
        for sent in cons_exs:
            c_array = [char for char in sent]
            c_idx = []
            for c in c_array:
                c_idx.append(vocab_index.index_of(c))
            self.examples.append(c_idx[:-1])
            self.labels.append(c_idx[1:])
        for sent in vowel_exs:
            c_array = [char for char in sent]
            c_idx = []
            for c in c_array:
                c_idx.append(vocab_index.index_of(c))
            self.examples.append(c_idx[:-1])
            self.labels.append(c_idx[1:])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = torch.tensor(one_hot_encoder(self.examples[idx], self.num_uni_char))
        label = torch.tensor(self.labels[idx])
        return example, label


class RNN(nn.Module):
    def __init__(self, vocab_index, num_hidden, num_layers, drop_prob):
        super().__init__()

        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.vocab_index = vocab_index

        self.lstm = nn.LSTM(len(vocab_index), num_hidden, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc_linear = nn.Linear(num_hidden, len(self.vocab_index))

    def forward(self, x, hidden):

        lstm_output, hidden = self.lstm(x, hidden)
        drop_output = self.dropout(lstm_output)
        drop_output = drop_output.contiguous().view(-1,self.num_hidden)
        final_out = self.fc_linear(drop_output)
        return final_out, hidden

    def hidden_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.num_hidden),
                torch.zeros(self.num_layers, batch_size, self.num_hidden))


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

    print(f'Dataset size: {len(train_data)*20}')
    batch_size = 100
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    HIDDEN_DIM = 96
    NUM_LAYERS = 3
    DROP_PROB = 0.5


    rnn = RNN(vocab_index, HIDDEN_DIM, NUM_LAYERS, DROP_PROB)

    total_param = []
    for p in rnn.parameters():
        total_param.append(int(p.numel()))

    print(f'Total model parameters: {sum(total_param)}')

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    batch_size = 100
    seq_len = len(train_data[0][0])
    tracker = 0
    for i in range(epochs):
        hidden = rnn.hidden_state(batch_size)
        for x, y in train_dataloader:
            tracker += 1
            hidden = tuple([state.data for state in hidden])
            rnn.zero_grad()
            lstm_output, hidden = rnn.forward(x, hidden)
            loss = criterion(lstm_output, y.view(batch_size*seq_len))
            loss.backward()
            nn.utils.clip_grad_norm(rnn.parameters(),max_norm=5)
            optimizer.step()

        if tracker%25 == 0:
            val_hidden = rnn.hidden_state(batch_size)
            val_losses = []
            rnn.eval()
            for x, y in test_dataloader:
                val_hidden = tuple([state.data for state in hidden])
                lstm_output, val_hidden = rnn.forward(x, val_hidden)
                val_loss = criterion(lstm_output, y.view(batch_size * seq_len))
                nn.utils.clip_grad_norm(rnn.parameters(), max_norm=5)
                val_losses.append(val_loss.item())
            rnn.train()
            print(f'EPOCH: {i}, STEP: {tracker}, VAL LOSS: {val_loss.item()}')

    return RNNClassifier(rnn, vocab_index)

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
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

class TextDataset2(Dataset):
    def __init__(self, cons_exs, vocab_index):
        self.examples = []
        self.labels = []
        self.num_uni_char = len(vocab_index)
        for sent in cons_exs:
            c_array = [char for char in sent]
            c_idx = []
            for c in c_array:
                c_idx.append(vocab_index.index_of(c))
            self.examples.append(c_idx[:-1])
            self.labels.append(c_idx[1:])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = torch.tensor(one_hot_encoder(self.examples[idx], self.num_uni_char))
        label = torch.tensor(self.labels[idx])
        return example, label

class RNNLanguageModel(LanguageModel):
    def __init__(self, rnn, vocab_indexer):
        self.rnn = rnn
        self.vocab_indexer = vocab_indexer
        self.voc_size = len(vocab_indexer)


    def get_next_char_log_probs(self, context):
        self.rnn.eval()
        hidden = self.rnn.hidden_state(1)
        hidden = tuple([state.data for state in hidden])
        c_array = [char for char in context]
        c_idx = []
        for c in c_array:
            c_idx.append(self.vocab_indexer.index_of(c))
        x = c_idx
        x = torch.tensor(one_hot_encoder(x, len(self.vocab_indexer)))
        lstm_output, hidden = self.rnn.forward(x.view(1, len(c_idx), len(self.vocab_indexer)), hidden)
        next_char_logit = torch.log_softmax(lstm_output.data[-1],0).numpy()
        return next_char_logit

    def get_log_prob_sequence(self, next_chars, context):
        #Encode the context
        self.rnn.eval()
        hidden = self.rnn.hidden_state(1)
        hidden = tuple([state.data for state in hidden])
        c_array = [char for char in context]
        c_idx = []
        for c in c_array:
            c_idx.append(self.vocab_indexer.index_of(c))
        x = c_idx
        c_array = [char for char in context]
        c_idx = []
        for c in c_array:
            c_idx.append(self.vocab_indexer.index_of(c))
        y = c_idx
        log_probs = []
        for c in y:
            log_probs.append(self.get_next_char_log_probs(x)[self.vocab_indexer.index_of(c)])
            x.append(c)
        log_prob_seq = sum(log_probs)
        print(f'log prob seq: {log_prob_seq}')
        return log_prob_seq


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    seq_len = 20
    train_text = [char for char in train_text]
    train_text = np.array(train_text).reshape((int(len(train_text)/seq_len),seq_len))

    dev_text = [char for char in dev_text]
    dev_text = np.array(dev_text).reshape((int(len(dev_text)/seq_len),seq_len))


    print(train_text[0])
    print(dev_text[0])
    train_data = TextDataset2(train_text, vocab_index)
    test_data = TextDataset2(dev_text, vocab_index)

    print(f'Dataset size: {len(train_data) * seq_len}')
    batch_size = 100

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    DROP_PROB = 0.5

    rnn = RNN(vocab_index, HIDDEN_DIM, NUM_LAYERS, DROP_PROB)

    total_param = []
    for p in rnn.parameters():
        total_param.append(int(p.numel()))

    print(f'Total model parameters: {sum(total_param)}')

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 20
    tracker = 0
    seq_len = len(train_data[0][0])
    for i in range(epochs):
        hidden = rnn.hidden_state(batch_size)
        for x, y in train_dataloader:
            tracker += 1
            hidden = tuple([state.data for state in hidden])
            rnn.zero_grad()
            lstm_output, hidden = rnn.forward(x, hidden)
            loss = criterion(lstm_output, y.view(batch_size * seq_len))
            loss.backward()
            nn.utils.clip_grad_norm(rnn.parameters(), max_norm=5)
            optimizer.step()

        if tracker % 25 == 0:
            val_hidden = rnn.hidden_state(1)
            val_losses = []
            rnn.eval()
            for x, y in test_dataloader:
                val_hidden = tuple([state.data for state in val_hidden])
                lstm_output, val_hidden = rnn.forward(x, val_hidden)
                val_loss = criterion(lstm_output, y.view(1 * seq_len))
                nn.utils.clip_grad_norm(rnn.parameters(), max_norm=5)
                val_losses.append(val_loss.item())
            rnn.train()
            print(f'EPOCH: {i}, STEP: {tracker}, VAL LOSS: {val_loss.item()}')

    return RNNLanguageModel(rnn, vocab_index)
