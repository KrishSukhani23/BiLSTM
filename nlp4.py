from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn as nnlower
import torch.nn.functional as F
import torch.optim as optim
import random
torch.manual_seed(0)
random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


def read(input_file_path):
    data = []
    temp_data = []
    tag_data = []
    temp_tag = []
    file1 = open(input_file_path, 'r')
    individual_lines = file1.readlines()
    for line in individual_lines:
        if len(line.strip()) == 0:
            data.append(temp_data)
            tag_data.append(temp_tag)
            temp_data = []
            temp_tag = []
        else:
            parts = line.split(' ')
            temp_data.append(parts[1])
            temp_tag.append(parts[2][:-1])
    data.append(temp_data)
    tag_data.append(temp_tag)
    return data, tag_data


x_train, y_train = read('./data/train')
x_dev, y_dev = read('./data/dev')


def read_test(input_file_path):
    data = []
    temp_data = []
    file1 = open(input_file_path, 'r')
    individual_lines = file1.readlines()
    for line in individual_lines:
        if len(line.strip()) == 0:
            data.append(temp_data)
            temp_data = []
        else:
            parts = line.strip().split(' ')
            temp_data.append(parts[1])
    data.append(temp_data)
    return data


x_test = read_test('./data/test')


def create_dictionaries(all_sentences, word2idx):
    vocab = set()
    for sent in all_sentences:
        for word in sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                vocab.add(word)
    return word2idx, vocab
# def create_tag_dictionaries(tags, tag2idx):
#     all_tags = set()
#     for tag in tags:
#         for item in tag:
#             if item not in tag2idx:
#                 tag2idx[item] = len(tag2idx)
#                 all_tags.add(item)
#     return tag2idx, all_tags


tag2idx = {'B-ORG': 0,
           'O': 1,
           'B-MISC': 2,
           'B-PER': 3,
           'I-PER': 4,
           'B-LOC': 5,
           'I-ORG': 6,
           'I-MISC': 7,
           'I-LOC': 8}


def generate_vectors(sentences, indexes):
    data_vector = []
    for sentence in sentences:
        sentence_vector = []
        for word in sentence:
            sentence_vector.append(indexes[word])
        data_vector.append(sentence_vector)
    return data_vector


def generate_tag_vectors(tags, tag2idx):
    tags_vec = []
    i = 0
    while i < len(tags):
        temp_tags = []
        j = 0
        while j < len(tags[i]):
            label = tags[i][j]
            if label in tag2idx:
                temp_tags.append(tag2idx[label])
            else:
                temp_tags.append(tag2idx["<UNK>"])
            j += 1
        tags_vec.append(temp_tags)
        i += 1
    return tags_vec


# Create an index of all words in the dataset
word2idx, train_vocab = create_dictionaries(x_train, {})
train_dev_words, dev_vocab = create_dictionaries(x_dev, word2idx)
index_wrds, test_vocab = create_dictionaries(x_test, train_dev_words)
if '<PAD>' not in index_wrds:
    index_wrds['<PAD>'] = len(index_wrds)
if '<UNK>' not in index_wrds:
    index_wrds['<UNK>'] = len(index_wrds)
# Convert text data into numerical vectors
x_train_vector = generate_vectors(x_train, index_wrds)
x_test_vector = generate_vectors(x_test, index_wrds)
x_dev_vector = generate_vectors(x_dev, index_wrds)
# Create a dictionary of labels and convert label data into numerical vectors
# tag2idx, tags = create_tag_dictionaries(y_train, {})
# tags2idx_dev, tags_dev = create_tag_dictionaries(y_dev, tag2idx)
y_train_vector = generate_tag_vectors(y_train, tag2idx)
y_dev_vector = generate_tag_vectors(y_dev, tag2idx)


def class_weights(tag2idx, y_train, y_dev):
    cls_wt = {key: 0 for key in tag2idx}
    total_tags = 0
    for data in [y_train, y_dev]:
        for tags in data:
            for tag in tags:
                total_tags += 1
                cls_wt[tag] += 1

    class_wt = [max(1.0, round(math.log(0.35 * total_tags / cls_wt[key]), 2))
                for key in cls_wt]
    return torch.tensor(class_wt)


class_weight = class_weights(tag2idx, y_train, y_dev)
print(class_weight)


class BiLSTM_DataLoader(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        instance_of_x = torch.tensor(self.x_train[index])
        instance_of_y = torch.tensor(self.y_train[index])
        return instance_of_x, instance_of_y


class CustomCollator(object):
    def __init__(self, words_dictionary, tags):
        self.params = words_dictionary
        self.label = tags

    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        y_len = [len(y) for y in y_batch]
        x_len = [len(x) for x in x_batch]
        max_seq_length = max([len(s) for s in x_batch])
        each_batch = self.params['<PAD>'] * \
            np.ones((len(x_batch), max_seq_length))
        each_batch_tags = -1 * np.zeros((len(x_batch), max_seq_length))

        j = 0
        while j < len(x_batch):
            cur_len = len(x_batch[j])
            each_batch[j][:cur_len] = x_batch[j]
            each_batch_tags[j][:cur_len] = y_batch[j]
            j += 1

        each_batch_tags = torch.LongTensor(each_batch_tags)
        each_batch = torch.LongTensor(each_batch)

        each_batch_tags = Variable(each_batch_tags)
        each_batch = Variable(each_batch)

        return each_batch, each_batch_tags, x_len, y_len


class BiLSTM_TestLoader(Dataset):
    def __init__(self, x_test):
        self.x_test = x_test

    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, index):
        instance_of_x = torch.tensor(self.x_test[index])
        return instance_of_x


class CustomTestCollator(object):
    def __init__(self, words_dict, tags):
        self.params = words_dict
        self.tags = tags

    def __call__(self, batch):
        x_batch = batch
        len_val = [len(val) for val in x_batch]
        # y_len = [len(y) for y in yy]
        max_seq_len = max([len(s) for s in x_batch])
        each_batch = self.params['<PAD>']*np.ones((len(x_batch), max_seq_len))
        # batch_tagss = -1*np.zeros((len(x_batch), max_seq_len))
        for j in range(len(x_batch)):
            cur_len = len(x_batch[j])
            each_batch[j][:cur_len] = x_batch[j]
            # batch_tagss[j][:cur_len] = yy[j]

        each_batch = torch.LongTensor(each_batch)
        each_batch = Variable(each_batch)

        return each_batch, len_val
# class CustomTestCollator(object):
#     def __init__(self, words_dictionary):
#         self.words_dictionary = words_dictionary
#     def __call__(self, batch):
#         x_batch = batch
#         len_val = [len(val) for val in x_batch]
#         max_length = max(len_val)
#         padded_batch = [self.words_dictionary.pad_sequence(x, max_length) for x in x_batch]
#         batch_final = torch.tensor(padded_batch)
#         return batch_final, len_val


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, linear_out_dim, hidden_dim, lstm_layers,
                 bidirectional, dropout_val, tag_size):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(
            hidden_dim*2 if bidirectional else hidden_dim, linear_out_dim)
        self.dropout = nn.Dropout(dropout_val)
        self.elu = nn.ELU(alpha=0.01)
        self.classifier = nn.Linear(linear_out_dim, tag_size)

    def forward(self, inputs, lengths):
        embeddings = self.embedding(inputs)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths,
                                                              batch_first=True,
                                                              enforce_sorted=False)
        packed_output, _ = self.LSTM(packed_embeddings)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.elu(output)
        output = self.classifier(output)
        return output


# Define hyperparameters
EMBEDDING_DIM = 100
NUM_LSTM_LAYERS = 1
LSTM_HIDDEN_DIM = 256
LSTM_DROPOUT = 0.33
LINEAR_OUTPUT_DIM = 128
# Define BiLSTM model
model = BiLSTM(len(index_wrds), EMBEDDING_DIM, LINEAR_OUTPUT_DIM,
               LSTM_HIDDEN_DIM, NUM_LSTM_LAYERS, True, LSTM_DROPOUT,
               len(tag2idx))
model.to(device)

# Define DataLoader and collator
load_train_data = BiLSTM_DataLoader(x_train_vector, y_train_vector)
collator = CustomCollator(index_wrds, tag2idx)
dataloader = DataLoader(dataset=load_train_data,
                        batch_size=32, drop_last=True, collate_fn=collator)

# Define loss function, optimizer, and number of epochs
class_weight = class_weights(tag2idx, y_train, y_dev)
class_weight_tensor = torch.FloatTensor(class_weight).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
num_epochs = 450

# Train the BiLSTM model
for epoch in range(num_epochs):
    train_loss = 0.0
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = model(input.to(device), input_len)
        output = output.view(-1, len(tag2idx))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(1)
    train_loss /= len(load_train_data)
    print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}')
torch.save(model.state_dict(), f'blstm1.pt')

dataloader_dev = DataLoader(
    dataset=BiLSTM_DataLoader(x_dev_vector, y_dev_vector),
    batch_size=1,
    shuffle=False,
    drop_last=True,
    collate_fn=CustomCollator(index_wrds, tag2idx)
)

idx2tag = {v: k for k, v in tag2idx.items()}
idx2word = {v: k for k, v in index_wrds.items()}
with open("dev1.out", 'w') as file:
    for data_dev, tag, len_dev, tag_data_len in dataloader_dev:
        pred = model(data_dev.to(device), len_dev)
        pred = pred.cpu().detach().numpy()
        tag = tag.detach().numpy()
        data_dev = data_dev.detach().numpy()
        pred = np.argmax(pred, axis=2)
        pred = pred.reshape((len(tag), -1))
        for i in range(len(data_dev)):
            for j in range(len(data_dev[i])):
                if data_dev[i][j] != 30290:
                    word = idx2word[data_dev[i][j]]
                    # gold = idx2tag[tag[i][j]]
                    op = idx2tag[pred[i][j]]
                    file.write(" ".join([str(j+1), word, op]))
                    file.write("\n")
            file.write("\n")
saved_model = BiLSTM(vocab_size=len(index_wrds), embedding_dim=100, linear_out_dim=128,
                     hidden_dim=256, lstm_layers=1, bidirectional=True, dropout_val=0.33,
                     tag_size=len(tag2idx))
saved_model.load_state_dict(torch.load("blstm1.pt"))
saved_model.to(device)

dataloader_test = DataLoader(dataset=BiLSTM_TestLoader(x_test_vector),
                             batch_size=1,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=CustomTestCollator(index_wrds, tag2idx))

rev_tag2idx = {v: k for k, v in tag2idx.items()}
rev_vocab_dict = {v: k for k, v in index_wrds.items()}
res = []
file = open("test1.out", 'w')
for test_data, test_data_len in dataloader_test:

    pred = saved_model(test_data.to(device), test_data_len)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    # label = label.detach().numpy()
    test_data = test_data.detach().numpy()
    pred = np.argmax(pred, axis=2)
    pred = pred.reshape((len(test_data), -1))

    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            if test_data[i][j] != 30290:
                word = rev_vocab_dict[test_data[i][j]]
                # gold = rev_tag2idx[label[i][j]]
                op = rev_tag2idx[pred[i][j]]
                res.append((word, op))
                file.write(" ".join([str(j + 1), word, op]))
                file.write("\n")
        file.write("\n")
file.close()


def embedding_matrix(word2idx, embedding_dictionary, dim):
    vocab_size = len(word2idx)
    emb_matrix = np.zeros((vocab_size, dim))
    unk_embedding = embedding_dictionary.get("<UNK>")
    i = 0
    while i < vocab_size:
        word = list(word2idx.keys())[i]
        idx = word2idx[word]
        emb = embedding_dictionary.get(
            word, embedding_dictionary.get(word.lower(), unk_embedding))
        emb_matrix[idx] = emb
        i += 1
    return emb_matrix


word2idx, train_vocab = create_dictionaries(x_train, {})
train_dev_words, dev_vocab = create_dictionaries(x_dev, word2idx)
word_to_index, test_vocab = create_dictionaries(x_test, train_dev_words)
if '<PAD>' not in word_to_index:
    word_to_index['<PAD>'] = len(word_to_index)
if '<UNK>' not in word_to_index:
    word_to_index['<UNK>'] = len(word_to_index)
glove = pd.read_csv('glove.6B.100d.gz', sep=" ",
                    quoting=3, header=None, index_col=0)
glove_dict = {word: vector for word, vector in glove.T.items()}
glove_vectors = np.array([glove_dict[word] for word in glove_dict])
glove_dict["<PAD>"] = np.zeros((100,), dtype="float64")
glove_dict["<UNK>"] = np.mean(
    glove_vectors, axis=0, keepdims=True).reshape(100,)
embedding_matrix = embedding_matrix(word_to_index, glove_dict, 100)
vocab_size = embedding_matrix.shape[0]
embedding_size = embedding_matrix.shape[1]
print(vocab_size, embedding_size)


class BiLSTM_glove(nn.Module):
    def __init__(self, vocab_size, embedding_dim, linear_out_dim, hidden_dim, lstm_layers,
                 bidirectional, dropout_val, tag_size, emb_matrix):
        super(BiLSTM_glove, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.embedding_dim = embedding_dim
        self.linear_out_dim = linear_out_dim
        self.tag_size = tag_size
        self.emb_matrix = emb_matrix
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix))
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*self.num_directions, linear_out_dim)
        self.dropout = nn.Dropout(dropout_val)
        self.elu = nn.ELU(alpha=0.01)
        self.classifier = nn.Linear(linear_out_dim, tag_size)

    def forward(self, sen, sen_len):
        batch_size = sen.shape[0]
        h_0, c_0 = (torch.zeros(self.lstm_layers * self.num_directions,
                                batch_size, self.hidden_dim).to(device),
                    torch.zeros(self.lstm_layers * self.num_directions,
                                batch_size, self.hidden_dim).to(device))

        embedded = self.embedding(sen).float()
        packed_embedded = pack_padded_sequence(
            embedded, sen_len, batch_first=True, enforce_sorted=False)
        output, _ = self.LSTM(packed_embedded, (h_0, c_0))
        output_unpacked, _ = pad_packed_sequence(output, batch_first=True)
        dropout = self.dropout(output_unpacked)
        lin = self.fc(dropout)
        pred = self.elu(lin)
        pred = self.classifier(pred)
        return pred


BiLSTM_model = BiLSTM_glove(
    len(word_to_index),
    EMBEDDING_DIM,
    LINEAR_OUTPUT_DIM,
    LSTM_HIDDEN_DIM,
    NUM_LSTM_LAYERS,
    True,
    LSTM_DROPOUT,
    len(tag2idx),
    embedding_matrix
).to(device)

BiLSTM_train = BiLSTM_DataLoader(x_train_vector, y_train_vector)
custom_collator = CustomCollator(word_to_index, tag2idx)

dataloader = DataLoader(
    dataset=BiLSTM_train,
    batch_size=8,
    drop_last=True,
    collate_fn=custom_collator
)

criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
criterion.requres_grad = True

optimizer = torch.optim.SGD(
    BiLSTM_model.parameters(),
    lr=0.1,
    momentum=0.9
)

scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
epochs = 65

for i in range(1, epochs+1):
    train_loss = 0.0
    for input, label, input_len, label_len in dataloader:
        optimizer.zero_grad()
        output = BiLSTM_model(input.to(device), input_len)
        output = output.view(-1, len(tag2idx))
        label = label.view(-1)
        loss = criterion(output, label.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input.size(1)

    train_loss = train_loss / len(dataloader.dataset)
    print(f'Epoch: {i} \tTraining Loss: {train_loss:.6f}')
torch.save(model.state_dict(), f'blstm2.pt')


dataloader_dev = DataLoader(
    dataset=BiLSTM_DataLoader(x_dev_vector, y_dev_vector),
    batch_size=8,
    shuffle=False,
    drop_last=True,
    collate_fn=CustomCollator(word_to_index, tag2idx)
)
rev_label_dict = {v: k for k, v in tag2idx.items()}
rev_vocab_dict = {v: k for k, v in word_to_index.items()}
res = []
with open("dev2.out", 'w') as file:
    for dev_data, label, dev_data_len, label_data_len in dataloader_dev:
        pred = BiLSTM_model(dev_data.to(device),
                            dev_data_len).cpu().detach().numpy()
        label = label.detach().numpy()
        dev_data = dev_data.detach().numpy()
        pred = np.argmax(pred, axis=2).reshape((len(label), -1))
        for i in range(len(dev_data)):
            for j in range(len(dev_data[i])):
                if dev_data[i][j] != 30290:
                    word = rev_vocab_dict[dev_data[i][j]]
                    # gold = rev_label_dict[label[i][j]]
                    op = rev_label_dict[pred[i][j]]
                    res.append((word, op))
                    file.write(" ".join([str(j + 1), word, op]))
                    file.write("\n")
            file.write("\n")

# !perl 'conll03eval.txt' < 'dev1.out'
# !perl 'conll03eval.txt' < 'dev2.out'
saved_model2 = BiLSTM(vocab_size=len(index_wrds), embedding_dim=100, linear_out_dim=128,
                      hidden_dim=256, lstm_layers=1, bidirectional=True, dropout_val=0.33,
                      tag_size=len(tag2idx))
saved_model2.load_state_dict(torch.load("blstm2.pt"))
saved_model2.to(device)

dataloader_test = DataLoader(dataset=BiLSTM_TestLoader(x_test_vector),
                             batch_size=1,
                             shuffle=False,
                             drop_last=True,
                             collate_fn=CustomTestCollator(index_wrds, tag2idx))

rev_tag2idx = {v: k for k, v in tag2idx.items()}
rev_vocab_dict = {v: k for k, v in index_wrds.items()}
res = []
file = open("test2.out", 'w')
for test_data, test_data_len in dataloader_test:

    pred = saved_model2(test_data.to(device), test_data_len)
    pred = pred.cpu()
    pred = pred.detach().numpy()
    # label = label.detach().numpy()
    test_data = test_data.detach().numpy()
    pred = np.argmax(pred, axis=2)
    pred = pred.reshape((len(test_data), -1))

    for i in range(len(test_data)):
        for j in range(len(test_data[i])):
            if test_data[i][j] != 30290:
                word = rev_vocab_dict[test_data[i][j]]
                # gold = rev_tag2idx[label[i][j]]
                op = rev_tag2idx[pred[i][j]]
                res.append((word, op))
                file.write(" ".join([str(j + 1), word, op]))
                file.write("\n")
        file.write("\n")
file.close()
