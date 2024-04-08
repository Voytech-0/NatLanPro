#!/usr/bin/env python
# coding: utf-8
import os

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import numpy as np
import random
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm


# In[2]:


random.seed(42)
np.random.seed(42)

save_path = "/scratch/s4296850/NLP/results_multihead"

if not os.path.exists(save_path):
    os.makedirs(save_path)


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


# dependent on the implementation of the word2index and index2word
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 80 + 2 # 80 words + SOS and EOS


# In[5]:


class Lang:
    """Class to store the vocabulary of a language and the mappings between words and indices."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[6]:


def readLangs(to_spanish=False):
    """Reads the lines from the csv file and creates the Lang instances for the input and output languages, as well as the pairs of sentences."""
    print("Reading lines...")

    # Read the cvs file
    file = pd.read_csv("/home3/s4296850/NLP/europarl-extract-master/corpora/cropped_europarl.csv")
    en = file["en"]
    es = file["es"]

    # Split every line into pairs

    # Reverse pairs, make Lang instances
    if to_spanish:
        pairs = list(zip(es, en))
        input_lang = Lang("es")
        output_lang = Lang("en")
    else:
        pairs = list(zip(en, es))
        input_lang = Lang("en")
        output_lang = Lang("es")
        
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    return input_lang, output_lang, pairs


# In[7]:


# input_lang, output_lang, pairs = readLangs()
# for pair in pairs:
#     input_lang.addSentence(pair[0])
#     output_lang.addSentence(pair[1])
# print(random.choice(pairs))


# In[8]:


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = readLangs()

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


# In[9]:


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, hidden = self.gru(self.dropout(self.embedding(x)))
        return output, hidden


# In[10]:


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_output)
            decoder_outputs.append(decoder_output)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        embedded_rel = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded_rel, hidden)
        output = self.out(output)
        return output, hidden


# In[11]:


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


# In[12]:


# just use torch multiheadAttention
class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=8):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_size, n_heads)

    def forward(self, query, keys):
        query = query.transpose(0, 1)  # [batch_size, 1, hidden_size] -> [1, batch_size, hidden_size]
        keys = keys.transpose(0, 1)    # [batch_size, seq_len, hidden_size] -> [seq_len, batch_size, hidden_size]
        context, attn_weights = self.attn(query, keys, keys)
        context = context.transpose(0, 1)  # [1, batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        return context, attn_weights
    


# In[13]:


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = attention_type
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


# In[14]:


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0
    for i, data in enumerate( dataloader):
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        if i % 500 == 0:
            torch.save(encoder.state_dict(), f"{save_path}/encoder_partial.pt")
            torch.save(decoder.state_dict(), f"{save_path}/decoder_partial.pt")

    return total_loss / len(dataloader)


# In[15]:


import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[16]:


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_every=100, plot_every=100):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(print_loss_avg)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)


# In[17]:


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


# In[18]:


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[19]:


hidden_size = 128
epochs = 20


# In[20]:


batch_size = 32
input_lang, output_lang, train_dataloader = get_dataloader(batch_size)


# In[ ]:


encoder = Encoder(input_lang.n_words, hidden_size).to(device)
# attention_type = BahdanauAttention(hidden_size)
attention_type = MultiheadAttention(hidden_size, 4)
decoder = AttnDecoder(hidden_size, output_lang.n_words, attention_type=attention_type).to(device)

train(train_dataloader, encoder, decoder, epochs, print_every=1000, plot_every=5000)

torch.save(encoder.state_dict(), f"{save_path}/encoder.pt")
torch.save(decoder.state_dict(), f"{save_path}/decoder.pt")


# In[ ]:


# encoder.eval()
# decoder.eval()
# evaluateRandomly(encoder, decoder)


# In[ ]:


# def bleuScore(encoder, decoder, input_lang, output_lang, pairs):
#
#     total_score = 0.0
#     with tqdm(total=len(pairs), desc="Calculating BLEU Score") as pbar:
#         for pair in pairs:
#             input_sentence = pair[0]
#             target_sentence = pair[1]
#             output_words, _ = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
#             output_sentence = ' '.join(output_words)
#             bleu = sentence_bleu([target_sentence.split()], output_sentence.split())
#             total_score += bleu
#             pbar.update(1)
#     average_bleu = total_score / len(pairs)
#     print('Average BLEU Score:', average_bleu)
#
# bleuScore(encoder, decoder, input_lang, output_lang, pairs)

