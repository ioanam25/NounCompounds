# import required module
import os
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch

directory = 'labels'

freq_label = {}
hits = {}
label_agrees = {}
confused = {}

index = {"Nicio varianta de mai sus": 0,
         "Proces + scopul procesului/entitatea care sufera procesul": 1,
         "Entitate + scopul ei": 2,
         "Posesor + ce posedeaza/detine": 3,
         "Activitate + cine face activitatea": 4,
         "Data evenimentului + eveniment": 5,
         "Eveniment + durata evenimentului": 6,
         "Entitate + atribut al entitatii": 7,
         "Entitate (intreg) + substanta/material/ingredient": 8,
         "Ce/cine este cauza + rezultat": 9,
         "Entitate + locatia unde se afla entitatea / locatia + ce se afla in locatie": 10,
         "Entitate/proces/rezultat + cauza/sursa": 11,
         "Parte/membru (detasabil) + entitatea din care face parte": 12,
         "Eveniment + cand se petrece evenimentul": 13,
         "Cine experimenteaza + sentimentul/senzatia/gandul experimentat": 14,
         "Beneficiar/cine primeste + de ce beneficiaza/ce primeste": 15,
         "Substantive provenite din verbe + substantive asupra carora se actioneaza": 16}

X = []
y = []

def replace_diacritics(word):
    word = word.replace("ă", "a")
    word = word.replace("â", "a")
    word = word.replace("î", "i")
    word = word.replace("ș", "s")
    word = word.replace("ț", "t")
    # print(word)
    word = word.lower()
    return word

for filename in os.listdir(directory):  # for each batch
    data = pd.read_csv('labels' + '/' + filename)
    for i in data['Input.C1']:
        words = []
        current = ""
        for j in range(len(i)):
            if i[j] == '_':
                words.append(current)
                current = ""
            else:
                current += i[j]
        current = replace_diacritics(current)
        words.append(current)
        if len(words) == 3:
            words[1] = words[2]
            words.pop()
        X.append(words)

    for label in data['Answer.category.label']:  # for each label in the batch
        y.append(index[label])

# for i in range(len(y)):
#     print(i, X[i], y[i])

num_label = len(y)
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(len(X_train), len(X_test))

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", output_hidden_states = True)

# token_embedding = {token: bert.get_input_embeddings()(torch.tensor(id))  for token, id in tokenizer.get_vocab().items()}
# print(len(token_embedding))
# print(token_embedding['[CLS]'])
# embedding_dim = len(token_embedding)

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT

    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids


    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids

    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings


# Getting embeddings for the target (our input words)
# word in all given contexts

texts = []
embedding = {}

for i in range(len(X)):
    texts.append(replace_diacritics(X[i][0]))
    texts.append(replace_diacritics(X[i][1]))

i = 1
for text in texts:
    print(text, i)
    i += 1
    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    print(tokenized_text)
    # Find the position 'text' in list of tokens
    if text not in tokenized_text:
        word_index = 1
    else:
        word_index = tokenized_text.index(text)

    # Get the embedding for bank
    word_embedding = list_token_embeddings[word_index]
    embedding[text] = word_embedding
    print(len(embedding[text]))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # bert embeddings size 768
        self.fc1 = nn.Linear(768 * 2, 768)
        self.fc2 = nn.Linear(768, 17)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Trainer():
    def __init__(self, net=None, optim=None, loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_functiongit
        self.train_loader = train_loader

    def train(self, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            for data in self.train_loader:
                self.optim.zero_grad()
                output = net(X_train)
                loss = self.loss_function(output, y_train)
                loss.backward()
                self.optim.step()
                epoch_loss += loss.item()
                epoch_steps += 1
            # average loss of epoch
            losses.append(epoch_loss / epoch_steps)
            print("epoch [%d]: loss %.3f" % (epoch + 1, losses[-1]))
        return losses


learning_rate = 0.01

net = Net()
opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_function = nn.CrossEntropyLoss()

trainer = Trainer(net=net, optim=opt, loss_function=loss_function, train_loader=train_loader)

losses = trainer.train(num_epochs)
output = net(X_train)

# let the maximum index be our predicted class
_, yh = torch.max(output, 1)

m = nn.Softmax(dim=1)
output = m(output)



