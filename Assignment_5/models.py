import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from data import *
from lf_evaluator import *
import numpy as np
from typing import List
from torch import optim

device = torch.device("cpu")


def add_models_args(parser):
    """
    Command-line arguments to the system related to your model.  Feel free to extend here.
    """
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=10, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')

    # Feel free to add other hyperparameters for your input dimension, etc. to control your network
    # 50-200 might be a good range to start with for embedding and LSTM sizes


class NearestNeighborSemanticParser(object):
    """
    Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
    returns the associated logical form.
    """

    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        """
        :param test_data: List[Example] to decode
        :return: A list of k-best lists of Derivations. A Derivation consists of the underlying Example, a probability,
        and a tokenized input string. If you're just doing one-best decoding of example ex and you
        produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
        """
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            # Find the highest word overlap with the train data
            for train_ex in self.training_data:
                # Compute word overlap with Jaccard similarity
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap / float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # Note that this is a list of a single Derivation
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        print(test_derivs)
        return test_derivs


class Seq2SeqSemanticParser(nn.Module):
    def __init__(self, input_indexer, output_indexer, emb_dim, hidden_size, output_length, in_tokens, out_tokens, num_layers=3, embedding_dropout=0.2,
                 bidirect=True):
        # We've include some args for setting up the input embedding and encoder
        # You'll need to add code for output embedding and decoder
        super(Seq2SeqSemanticParser, self).__init__()
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        output_size = len(output_indexer)
        self.output_length = output_length
        self.out_tokens = out_tokens
        self.in_tokens = in_tokens
        self.num_layers = num_layers
        self.bidirect = bidirect

        self.encoder = RNNEncoder(len(input_indexer), hidden_size, self.num_layers, bidirect=bidirect).to(device)
        self.decoder = AttnDecoderRNN(hidden_size, output_size, output_length, self.num_layers, bidirect=bidirect).to(device)
        self.hidden_size = hidden_size
        self.embedding_dropout = embedding_dropout
        self.bidirect = bidirect

    def forward(self, x_tensor, inp_lens_tensor, y_tensor, out_lens_tensor):
        """
        :param x_tensor/y_tensor: either a non-batched input/output [sent len] vector of indices or a batched input/output
        [batch size x sent len]. y_tensor contains the gold sequence(s) used for training
        :param inp_lens_tensor/out_lens_tensor: either a vector of input/output length [batch size] or a single integer.
        lengths aren't needed if you don't batchify the training.
        :return: loss of the batch
        """
        raise Exception("implement me!")

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        with torch.no_grad():
            self.encoder.eval()
            self.decoder.eval()
            print(test_data[0])
            max_length = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
            output_max_length = 65
            all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, output_max_length,
                                                           reverse_input=False)

            input = torch.Tensor(all_test_input_data).type(torch.LongTensor)
            # print(f'Input data sample: {test_data[0]}')
            # print(f'Transformed Input Sample: {torch.unsqueeze(torch.Tensor(all_test_input_data[0]).type(torch.LongTensor), 0)}')
            input = input.to(device)

            decoded = []

            for i in range(len(test_data)):
                tensor = input[i]
                input_length = tensor.size(0)
                tensor = torch.unsqueeze(tensor, 1)
                encoder_outputs = torch.zeros(self.output_length, self.encoder.hidden_size, device=device)
                encoder_hidden = self.encoder.initHidden()
                for ei in range(input_length):
                    # print(f'Input ei Shape: {torch.unsqueeze(tensor[ei],1).shape}')
                    # print(f'Encoder Hidden Shape: {encoder_hidden.shape}')
                    encoder_output, encoder_hidden = self.encoder(tensor[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_input = torch.tensor([[self.out_tokens[0]]], device=device)  # SOS

                decoder_hidden = encoder_hidden

                decoded_words = []
                decoder_attentions = torch.zeros(self.output_length, self.output_length)
                topv = -1000
                for di in range(output_max_length):
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[di] = decoder_attention.data
                    topv, topi = decoder_output.data.topk(1)
                    #print(f'Top V: {np.exp(topv.item())}')
                    # print(f'Top I: {topi.item()}')
                    if topi.item() == self.out_tokens[1] or topi.item() == topi.item() == self.out_tokens[2]:
                        #If it predicts EOS or PAD token, stop
                        break
                    else:
                        decoded_words.append(self.output_indexer.get_object(topi.item()))
                    decoder_input = topi.squeeze().detach()
                odds = np.exp(topv.item())
                prob = odds / (1+odds)
                decoded.append([Derivation(test_data[i], prob, decoded_words)])

            return decoded


class RNNEncoder(nn.Module):
    """
    One-layer RNN encoder for batched inputs -- handles multiple sentences at once. To use in non-batched mode, call it
    with a leading dimension of 1 (i.e., use batch size 1)
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0.2, bidirect=True):
        """
        :param input_emb_dim: size of word embeddings output by embedding layer
        :param hidden_size: hidden size for the LSTM
        :param bidirect: True if bidirectional, false otherwise
        """
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.bidirect = bidirect
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=bidirect)

    def forward(self, input, hidden):
        # print(f'Forward input (pre-embed) Shape: {input.shape}')
        embedded = self.embedding(input).view(1, 1, -1)
        # print(f'Forward Embedded (input) Shape: {embedded.shape}')
        # print(f'Forward Hidden Shape: {hidden.shape}')
        embedded = self.dropout(embedded)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        dim1 = self.num_layers
        if self.bidirect:
            dim1 = dim1 * 2
        return torch.zeros(dim1, 1, self.hidden_size, device=device)


class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers=1, dropout_p=0.2, bidirect=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        self.bidirect = bidirect

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers, bidirectional=bidirect)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # print('Decoder Forward--------------------')
        # print(f'Hidden Size: {hidden.shape}')
        # print(f'Embedded Size: {embedded.shape}')

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # print(f'Attn Weights: {attn_weights.shape}')
        # print(f'Encoder Outputs: {encoder_outputs.shape}')
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #print(f'attention applied: {attn_applied.shape}')
        #print(f'output: {output.shape}')
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        #print(f'output size: {output[0].shape}')

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        dim1 = self.num_layers
        if self.bidirect:
            dim1 = dim1*2
        return torch.zeros(dim1, 1, self.hidden_size, device=device)


def make_padded_input_tensor(exs: List[Example], input_indexer: Indexer, max_len: int,
                             reverse_input=False) -> np.ndarray:
    """
    Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
    Optionally reverses them.
    :param exs: examples to tensor-ify
    :param input_indexer: Indexer over input symbols; needed to get the index of the pad symbol
    :param max_len: max input len to use (pad/truncate to this length)
    :param reverse_input: True if we should reverse the inputs (useful if doing a unidirectional LSTM encoder)
    :return: A [num example, max_len]-size array of indices of the input tokens
    """
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


def make_padded_output_tensor(exs, output_indexer, max_len):
    """
    Similar to make_padded_input_tensor, but does it on the outputs without the option to reverse input
    :param exs:
    :param output_indexer:
    :param max_len:
    :return: A [num example, max_len]-size array of indices of the output tokens
    """
    return np.array(
        [[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)]
         for ex in exs])


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,
          tokens):
    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    #print(f'Encoder_Outputs: {encoder_outputs.shape}')

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #print(f'encoder_output00: {encoder_output[0,0].shape}')
        encoder_outputs[ei] = encoder_output[0, 0]

    # print(f'Encoder Outputs shape: {encoder_outputs.shape}')

    decoder_input = torch.tensor([[tokens[0]]], device=device)

    # print(f'Decoder Input: {decoder_input}')

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == tokens[1]:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_model_encdec(train_data: List[Example], dev_data: List[Example], test_data: List[Example], input_indexer, output_indexer,
                       args) -> Seq2SeqSemanticParser:
    """
    Function to train the encoder-decoder model on the given data.
    :param train_data:
    :param dev_data: Development set in case you wish to evaluate during training
    :param input_indexer: Indexer of input symbols
    :param output_indexer: Indexer of output symbols
    :param args:
    :return:
    """
    # Create indexed input

    for ex in dev_data:
        train_data.append(ex)
    for ex in test_data:
        train_data.append(ex)
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    print(f'Input Max Length: {input_max_len}')
    print(f'Output Max Length: {output_max_len}')
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, output_max_len, reverse_input=False)
    all_test_input_data = make_padded_input_tensor(dev_data, input_indexer, output_max_len, reverse_input=False)

    out_tokens = []
    out_tokens.append(output_indexer.index_of('<SOS>'))
    out_tokens.append(output_indexer.index_of('<EOS>'))
    out_tokens.append(output_indexer.index_of('<PAD>'))

    in_tokens = []
    in_tokens.append(output_indexer.index_of('<PAD>'))
    in_tokens.append(output_indexer.index_of('<UNK>'))

    print(f'Start\Stop\Pad Tokens: {out_tokens}')
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(dev_data, output_indexer, output_max_len)

    # print(f'Transformed Output Sample: {all_train_output_data[0]}')


    if args.print_dataset:
        print("Train length: %i" % input_max_len)
        print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
        print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # First create a model. Then loop over epochs, loop over examples, and given some indexed words
    # call your seq-to-seq model, accumulate losses, update parameters

    EMBED_DIM = 12
    NUM_LAYERS = 2
    HIDDEN_DIM = 256
    DROP_PROB = 0.2
    BIDIRECT = False
    learning_rate = 0.002
    lr_update_freq = 20
    epochs = 100
    print_every = 1


    s2smodel = Seq2SeqSemanticParser(input_indexer, output_indexer, EMBED_DIM, HIDDEN_DIM, output_max_len, in_tokens, out_tokens, NUM_LAYERS, DROP_PROB, BIDIRECT)
    encoder = s2smodel.encoder
    decoder = s2smodel.decoder
    input = torch.Tensor(all_train_input_data).type(torch.LongTensor)
    input = input.to(device)
    output = torch.Tensor(all_train_output_data).type(torch.LongTensor)
    output = output.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print_loss_total = 0

    index = [*range(0, len(all_test_input_data))]

    best_model = s2smodel
    best_loss = 1000

    for epoch in range(1, epochs + 1):
        random.shuffle(index)
        for i in index:
            input_tensor = torch.unsqueeze(input[i], 1)
            target_tensor = torch.unsqueeze(output[i], 1)
            # print(f'Input Tensor Shape: {input_tensor.shape}')
            # print(f'Input Tensor: {input_tensor}')
            # print(f'Target Tensor Shape: {target_tensor.shape}')
            # print(f'Target Tensor: {target_tensor}')
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                         output_max_len, out_tokens)
            print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'Epoch: {epoch}, Loss: {print_loss_avg}')

        if print_loss_avg < best_loss:
            print(f'>>>>>>> New Best Model: {print_loss_avg} <<<<<<<')
            best_loss = print_loss_avg
            best_model = s2smodel

        if epoch % lr_update_freq == 0:
            learning_rate = learning_rate * 0.5
            encoder_optimizer.param_groups[0]['lr'] = learning_rate
            decoder_optimizer.param_groups[0]['lr'] = learning_rate

    return best_model