#
#
#


import os
import random
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from train import train_round
from tqdm import tqdm

from agents import TSAgent
import obverter
from data import OrganismsDataset
from utils import *

seed = 365
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# run it on GPU if cuda available (recommended)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of games played in each round (mini-batch size)
batch_size = 24

# number of symbols (words) available to each agent to use
vocab_size = 2

# length of maximum sentence they can compose
max_sen_len = 10

# number of epochs/rounds
rounds = 400

# instantiate the agents
agent_a = TSAgent(vocab_size=vocab_size).to(device)
agent_b = TSAgent(vocab_size=vocab_size).to(device)

# negative log liklihood loss
criterion = nn.NLLLoss()

# optimizers
optimizer_a_gen = optim.Adam([p for p in agent_a.parameters() if p.requires_grad], lr=6e-4)
optimizer_b_gen = optim.Adam([p for p in agent_b.parameters() if p.requires_grad], lr=6e-4)


training_set = OrganismsDataset(path='./organisms/', return_pair=True, n_pairs=1000)
testing_set = OrganismsDataset('./organisms/test/', return_pair=True, n_pairs=1000)


agent1_accuracy_history = []
agent1_message_length_history = []
agent1_loss_history = []

os.makedirs('checkpoints', exist_ok=True)

for round in tqdm(range(rounds)):
    print("********** round %d **********" % round)
    round_accuracy, round_loss, round_sentence_length = train_round(agent_a, agent_b, optimizer_b_gen, max_sen_len, vocab_size)
    print_round_stats(round_accuracy, round_loss, round_sentence_length)

    agent1_accuracy_history.append(round_accuracy)
    agent1_message_length_history.append(round_sentence_length / 10)
    agent1_loss_history.append(round_loss)

    round += 1
    print("replacing roles")
    print("********** round %d **********" % round)

    round_accuracy, round_loss, round_sentence_length = train_round(agent_b, agent_a, optimizer_a_gen, max_sen_len, vocab_size)
    print_round_stats(round_accuracy, round_loss, round_sentence_length)

    t = list(range(len(agent1_accuracy_history)))
    plt.plot(t, agent1_accuracy_history, label="Accuracy")
    plt.plot(t, agent1_message_length_history, label="Message length (/10)")
    plt.plot(t, agent1_loss_history, label="Training loss")

    plt.xlabel('# Rounds')
    plt.legend()
    plt.savefig("graph.png")
    plt.clf()

    if round % 50 == 0:
        torch.save(agent_a.state_dict(), os.path.join('checkpoints', 'agent1-%d.ckp' % round))
        torch.save(agent_b.state_dict(), os.path.join('checkpoints', 'agent2-%d.ckp' % round))



# import torch
# import random
# from torch.nn import NLLLoss
# from torch.utils.data import DataLoader, random_split
# import matplotlib.pyplot as plt
# from agents import TalkingAgent
# from data import OrganismsDataset, is_edible, categorize
# from datetime import datetime
# from visdom import Visdom
# import numpy as np
# import os
# import torchvision
# from torch.nn.functional import softmax, log_softmax
# from utils import *
# import torch.optim as optim
#
#
# SEED = 1254
# random.seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# n_batches = 16
# sentence_len = 8
#
# # instantiate the agents
# agent_a = TalkingAgent(event_vector_size=sentence_len, batch_size=n_batches).to(device)
# agent_b = TalkingAgent(event_vector_size=sentence_len, batch_size=n_batches).to(device)
#
# # loss function
# criterion = nn.CosineEmbeddingLoss()
#
# # optimizers
# optimizer_a = optim.SGD(agent_a.parameters(), lr=0.0001, momentum=0.9)
# optimizer_b = optim.SGD(agent_b.parameters(), lr=0.0001, momentum=0.9)
#
# # load data and split it to train and test sets
# training_set = OrganismsDataset(return_pair=True, n_pairs=2000)
# test_set = OrganismsDataset('./organisms/test/', return_pair=True, n_pairs=100)
# train_loader = DataLoader(training_set, batch_size=n_batches, pin_memory=True, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_set, batch_size=n_batches, pin_memory=True, shuffle=False, drop_last=True)
#
# # setup dashboard for monitoring the training process
# viz = Visdom()
# viz.line([0.], [0], win='agent A', opts=dict(title='loss (Agent A)'))
# viz.line([0.], [0], win='agent B', opts=dict(title='loss (Agent B)'))
# # viz.line([0.], [0], win='accuracy', opts=dict(title='accuracy'))
# samples = test_set.pick_samples_from_each_cat(3)
#
#
# def convert_to_chars(raw_sen):
#     return ''.join([chr(int(i) + 96) for i in raw_sen.tolist()[0]])
#
#
# def test(agent_a_, agent_b_, test_set_, samples_, round):
#     agent_a_.eval()
#     agent_b_.eval()
#
#     with torch.no_grad():
#         results = []
#         for cat, orgs in samples_.items():
#             for o in orgs:
#                 frame_zero, motion_field, kind, motion_type, _ = test_set_.preprocess(o)
#                 frame_zero = frame_zero.unsqueeze(0).to(device)
#                 motion_field = motion_field.unsqueeze(0).to(device)
#
#                 row = {
#                     'org_path': o,
#                     'round': round,
#                     'cat': cat,
#                     'kind': kind,
#                     'motion': motion_type,
#                     'agent_a_call': convert_to_chars(agent_a_(frame_zero, motion_field)[-1]),
#                     'agent_b_call': convert_to_chars(agent_b_(frame_zero, motion_field)[-1])
#                 }
#                 results.append(row)
#
#     return results
#
#
# def train(n_of_epochs=2000):
#     for epoch in range(n_of_epochs):  # loop over the dataset multiple times
#         running_loss_a = 0.0
#         running_loss_b = 0.0
#         print(f'############ round: {epoch} ############')
#         for i, data in enumerate(train_loader):
#             batch_n = epoch * len(train_loader) + i + 1
#
#             frame_sp = data[0][0]
#             motion_sp = data[0][1]
#             frame_ls = data[1][0]
#             motion_ls = data[1][1]
#             label = data[2]
#
#             # pick the speaker
#             if random.choice([True, False]):
#                 assigned_agent = 'a'
#                 speaker = agent_a
#                 listener = agent_b
#                 optimizer_sp = optimizer_a
#                 optimizer_ls = optimizer_b
#             else:
#                 assigned_agent = 'b'
#                 speaker = agent_b
#                 listener = agent_a
#                 optimizer_sp = optimizer_b
#                 optimizer_ls = optimizer_a
#
#             # set the optimizers params to zero
#             optimizer_sp.zero_grad()
#             optimizer_ls.zero_grad()
#
#             # speaker training
#             vis_emb, lin_emb, _ = speaker(frame_sp.to(device), motion_sp.to(device))
#
#             # turn off grad in Conv layer only in speaker model
#             # for p in speaker.Conv.parameters():
#             #     p.requires_grad = False
#             # for p in speaker.Vis_out.parameters():
#             #     p.requires_grad = False
#
#             loss_sp = criterion(vis_emb, lin_emb, torch.tensor(1, device=device))  # label should be always one for speaker's part
#             loss_sp.backward()
#             optimizer_sp.step()
#
#             if batch_n % 10 == 9:
#                 if assigned_agent == 'a':
#                     running_loss_a += loss_sp.item()
#                     # viz.line([loss_sp.item()], [batch_n], win='agent A', update='append')
#                     viz.line([running_loss_a/10], [batch_n], win='agent A', update='append')
#                 else:
#                     running_loss_b += loss_sp.item()
#                     # viz.line([loss_sp.item()], [batch_n], win='agent B', update='append')
#                     viz.line([running_loss_b / 10], [batch_n], win='agent B', update='append')
#
#             with torch.no_grad():
#                 _, _, sentence_sp = speaker(frame_sp.to(device), motion_sp.to(device))
#             print(f'++++++agent {assigned_agent}+++++\n\n', sentence_sp)
#
#             # listener training
#             vis_emb, lin_emb, _ = listener(frame_ls.to(device), motion_ls.to(device), aux_sen=sentence_sp)
#
#             # turn off grad in language module only in listener model
#             for p in listener.EventEmbedder.parameters():
#                 p.requires_grad = False
#             # for p in listener.Lang_out.parameters():
#             #     p.requires_grad = False
#
#
#             loss_ls = criterion(vis_emb, lin_emb, torch.tensor(label, device=device))
#             loss_ls.backward()
#             optimizer_ls.step()
#
#             if batch_n % 10 == 9:
#                 if assigned_agent == 'b':
#                     # running_loss_b += loss_ls.item()
#                     # viz.line([loss_ls.item()], [batch_n], win='agent B', update='append')
#                     viz.line([running_loss_b / 10], [batch_n], win='agent B', update='append')
#                 else:
#                     # running_loss_b += loss_sp.item()
#                     # viz.line([loss_ls.item()], [batch_n], win='agent A', update='append')
#                     viz.line([running_loss_a / 10], [batch_n], win='agent A', update='append')
#
#                 running_loss_a = 0.0
#                 running_loss_b = 0.0
#
#             for p in speaker.Vision.parameters():
#                 p.requires_grad = True
#             for p in listener.EventEmbedder.parameters():
#                 p.requires_grad = True
#
#             # for p in speaker.Vis_out.parameters():
#             #     p.requires_grad = True
#             # for p in speaker.Lang_out.parameters():
#             #     p.requires_grad = True
#
#             # if i % 20 == 19:  # print every 20 mini-batches
#             #     print('[%d, %5d] agent A loss: %.3f' %
#             #           (epoch + 1, i + 1, running_loss_a / 20))
#             #     print('[%d, %5d] agent B loss: %.3f' %
#             #           (epoch + 1, i + 1, running_loss_b / 20))
#             #
#             #     running_loss_a = 0.0
#             #     running_loss_b = 0.0
#
#         if epoch % 50 == 0:
#             torch.save(agent_a.state_dict(), os.path.join('checkpoints', f'test-agentA-{epoch + 1}.ckp'))
#             torch.save(agent_b.state_dict(), os.path.join('checkpoints', f'test-agentB-{epoch + 1}.ckp'))
#
#         #     loss = loss_fn(output.to(device), torch.tensor(labels).to(device))
#         #     loss.backward()
#         #     optimizer.step()
#         #
#         #     # predictions = torch.round(probs)
#         #     # correct_vector = [1 for p, q in zip(predictions.tolist(), labels) if p == q]
#         #     # n_correct = sum(correct_vector)
#         #     # print("batch accuracy", n_correct / len(frame_zero))
#         #     # print("batch loss", loss.item())
#         #
#         #     batch_n = epoch * len(train_loader) + i + 1
#         #
#         #     viz.line([loss.item()], [batch_n], win='train_loss', update='append')
#         #     # viz.line([n_correct / len(frame_zero)], [batch_n], win='accuracy', update='append')
#         #
#         # if epoch % 20 == 0:
#         #     torch.save(silent_agent.state_dict(), os.path.join('checkpoints', f'test-multi-{epoch + 1}.ckp'))
#
#
# # def evaluate():
# #     data_test_iter = iter(test_loader)
# #
# #     images, mfs, kind, motion_type, orgs = next(data_test_iter)
# #
# #     # print images
# #     # imshow(torchvision.utils.make_grid(images))
# #     # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# #     silent_agent.load_state_dict(torch.load(r'checkpoints/test-multi-21.ckp'))
# #
# #     silent_agent.eval()
# #
# #     for layer in silent_agent.children():
# #         if hasattr(layer, '__len__'):
# #             for s_layer in range(len(layer)):
# #                 print(s_layer)
# #                 if type(layer[s_layer]) == nn.BatchNorm2d:
# #                     layer[s_layer].train()
# #                     layer[s_layer].momentum = 0.01
# #
# #     with torch.no_grad():
# #         correct = 0
# #         total = 0
# #         for data in test_loader:
# #             images, mfs, kind, motion_type, orgs = data
# #             labels = [categorize(k, m) for k, m in zip(kind, motion_type)]
# #             labels = torch.tensor(labels).to(device)
# #             outputs = silent_agent(images.to(device), mfs.to(device))
# #
# #             # log_, probs = log_softmax(outputs, dim=1), softmax(outputs, dim=1)[:, 1]
# #
# #             _, predictions = torch.max(outputs, 1)
# #             # correct_vector = [1 for p, q in zip(predictions.tolist(), labels) if p == q]
# #             total += labels.size(0)
# #             correct += (predictions == labels).sum().item()
# #             # n_correct = sum(correct_vector)
# #             # print(f'{n_correct} out of {len(predictions)}')
# #             # print("batch accuracy", n_correct / len(predictions))
# #
# #     print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}')
# #
# #
# if __name__ == '__main__':
#     pass
#     agent_a_ = TalkingAgent(event_vector_size=sentence_len, batch_size=1).to(device)
#     agent_b_ = TalkingAgent(event_vector_size=sentence_len, batch_size=1).to(device)
#
#     agent_a_.load_state_dict(torch.load('checkpoints/test-agentA-551.ckp'))
#     agent_b_.load_state_dict(torch.load('checkpoints/test-agentB-551.ckp'))
#
#     print(test(agent_a_, agent_b_, test_set, samples, 1))
#
#     # train()
# #     # evaluate()
#
# # from io import open
# # import unicodedata
# # import string
# # import re
# # import random
# #
# # import torch
# # import torch.nn as nn
# # from torch import optim
# # import torch.nn.functional as F
# #
# # from agents import SilentAgent
# #
# #
# #
# #
# # class TalkingAgent(nn.Module):
# #     def __init__(
# #             self,
# #             enc_input_size,
# #             enc_embedding_size,
# #             enc_hidden_size,
# #             enc_num_layers,
# #             dec_input_size,
# #             dec_embedding_size,
# #             dec_hidden_size,
# #             dec_output_size,
# #             dec_num_layers
# #     ):
# #         super().__init__()
# #         self.Conv = nn.Sequential(
# #             # 1st Conv layer
# #             nn.Conv2d(1, 128, 3, stride=2),
# #             nn.ReLU(),
# #             # 2nd Conv layer
# #             nn.Conv2d(128, 64, 3, stride=2),
# #             nn.ReLU(),
# #             nn.MaxPool2d(2, stride=2),
# #             nn.BatchNorm2d(64, momentum=0.01),
# #             nn.Dropout(0.2),
# #             # 3rd Conv layer
# #             nn.Conv2d(64, 32, 3, stride=2),
# #             nn.ReLU(),
# #             # 4th Conv layer
# #             nn.Conv2d(32, 32, 3, stride=2),
# #             nn.ReLU(),
# #             nn.MaxPool2d(2, stride=2),
# #             nn.Dropout(0.2)
# #         )
# #
# #         # as cnn output (visual module output)
# #         self.VisualFC = nn.Sequential(
# #             nn.Linear(576, 128),
# #             nn.Sigmoid(),
# #         )
# #
# #         # language module
# #         self.MeaningEncoder = Encoder(
# #             input_size=enc_input_size,
# #             embedding_size=enc_embedding_size,
# #             hidden_size=enc_hidden_size,
# #             num_layers=enc_num_layers
# #         )
# #         self.MeaningDecoder = Decoder(
# #             input_size=dec_input_size,
# #             embedding_size=dec_embedding_size,
# #             hidden_size=dec_hidden_size,
# #             output_size=dec_output_size,
# #             num_layers=dec_num_layers
# #         )
# #         self.LangModule = Seq2Seq(
# #             encoder=self.MeaningEncoder,
# #             decoder=self.MeaningDecoder
# #         )
# #
# #     def forward(self, zero_frame, motion_field):
# #         # pass the frame 0 thru conv layer
# #         x_1 = self.Conv(zero_frame)  # size 32x32x3X3
# #         x_1 = x_1.view(-1, x_1.size(1) * x_1.size(2) * x_1.size(3))  # size 32x288
# #
# #         # pass the motion field thru conv layer
# #         x_2 = self.Conv(motion_field)  # size 32x32x3X3
# #         x_2 = x_2.view(-1, x_2.size(1) * x_2.size(2) * x_2.size(3))  # size 32x288
# #         # concatenate ConvFixed and ConvMotion outputs
# #         x = torch.cat((x_1, x_2), dim=-1)
# #
# #         # pass the concatenated outputs to the fully connected layers
# #         x = self.VisualFC(x)
# #         x = self.fc2(x)
# #         x = self.fc3(x)
# #
# #         return x
# #
# #
# # class Encoder(nn.Module):
# #     def __init__(self, input_size, embedding_size, hidden_size, num_layers):
# #         super(Encoder, self).__init__()
# #         self.dropout = nn.Dropout()
# #         self.hidden_size = hidden_size
# #         self.num_layers = num_layers
# #
# #         self.embedding = nn.Embedding(input_size, embedding_size)
# #
# #         self.rnn = nn.LSTM(
# #             embedding_size,
# #             hidden_size,
# #             num_layers,
# #             dropout=0.2
# #         )
# #
# #     def forward(self, x):
# #         # x shape: (seq_length, N) where N is batch size
# #         embedding = self.dropout(self.embedding(x))
# #         # print('original:::::::::::::::::::::::::::::\n', x)
# #         # print('embedding::::::::::::::::::::::::::::\n', self.embedding(x))
# #         # print('emb. size::::::::::::::::::::::::::::\n', self.embedding(x).size())
# #
# #         # embedding shape: (seq_length, N, embedding_size)
# #         _, (hidden, cell) = self.rnn(embedding)
# #         # outputs shape: (seq_length, N, hidden_size)
# #
# #         return hidden, cell
# #
# #
# # class Decoder(nn.Module):
# #     def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers):
# #         super(Decoder, self).__init__()
# #         self.dropout = nn.Dropout(0.2)
# #         self.hidden_size = hidden_size
# #         self.num_layers = num_layers
# #
# #         self.embedding = nn.Embedding(input_size, embedding_size)
# #         self.rnn = nn.LSTM(
# #             embedding_size,
# #             hidden_size,
# #             num_layers,
# #             dropout=0.2
# #         )
# #         self.fc = nn.Linear(
# #             hidden_size,
# #             output_size
# #         )
# #
# #     def forward(self, x, hidden, cell):
# #         # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
# #         # is 1 here because we are sending in a single word and not a sentence
# #         x = x.unsqueeze(0)
# #
# #         embedding = self.dropout(self.embedding(x))
# #         # embedding shape: (1, N, embedding_size)
# #
# #         outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
# #         # outputs shape: (1, N, hidden_size)
# #
# #         predictions = self.fc(outputs)
# #
# #         # predictions shape: (1, N, length_target_vocabulary) to send it to
# #         # loss function we want it to be (N, length_target_vocabulary) so we're
# #         # just gonna remove the first dim
# #         predictions = predictions.squeeze(0)
# #
# #         return predictions, hidden, cell
# #
# #
# # class Seq2Seq(nn.Module):
# #     def __init__(self, encoder, decoder):
# #         super(Seq2Seq, self).__init__()
# #         self.encoder = encoder
# #         self.decoder = decoder
# #
# #     def forward(self, source, target, teacher_force_ratio=0.5):
# #         batch_size = source.shape[1]
# #         target_len = target.shape[0]
# #         target_vocab_size = len(english.vocab)
# #
# #         outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
# #         # print('output size', outputs.size())
# #         print('german source size::::::', source.size()) # n_of_words X batch size
# #         hidden, cell = self.encoder(source)
# #
# #         # Grab the first input to the Decoder which will be <SOS> token
# #         x = target[0]
# #
# #         # print('>>>>>>>>>>>>>>>>>>>>>>>>', x, x.size())
# #
# #         for t in range(1, target_len):
# #             # Use previous hidden, cell as context from encoder at start
# #             output, hidden, cell = self.decoder(x, hidden, cell)
# #
# #             # Store next output prediction
# #             outputs[t] = output
# #
# #             # Get the best word the Decoder predicted (index in the vocabulary)
# #             best_guess = output.argmax(1)
# #
# #             # With probability of teacher_force_ratio we take the actual next word
# #             # otherwise we take the word that the Decoder predicted it to be.
# #             # Teacher Forcing is used so that the model gets used to seeing
# #             # similar inputs at training and testing time, if teacher forcing is 1
# #             # then inputs at test time might be completely different than what the
# #             # network is used to. This was a long comment.
# #             x = target[t] if random.random() < teacher_force_ratio else best_guess
# #
# #         return outputs
# #
# #
# # if __name__ == '__main__':
# #     import torch
# #     import torch.nn as nn
# #     import torch.optim as optim
# #     from torchtext.datasets import Multi30k  # German to English dataset
# #     from torchtext.data import Field, BucketIterator
# #     import numpy as np
# #     import spacy
# #     import random
# #     # from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
# #     import torch
# #     import spacy
# #     from torchtext.data.metrics import bleu_score
# #     import sys
# #
# #     # Loading Tokeniser in german and English
# #     spacy_ger = spacy.load('de_core_news_sm')
# #     spacy_eng = spacy.load('en_core_web_sm')
# #
# #
# #     # Tokenization of German Language
# #     def tokenize_ger(text):
# #         return [tok.text for tok in spacy_ger.tokenizer(text)]
# #
# #
# #     # Tokenization of English Language
# #     def tokenize_eng(text):
# #         return [tok.text for tok in spacy_eng.tokenizer(text)]
# #
# #
# #     # Applyling Tokenization , lowercase and special Tokens for preprocessing
# #     german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')
# #
# #     english = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')
# #
# #     # Dwonloading Dataset and storing them
# #     train_data, valid_data, test_data = Multi30k.splits(
# #         exts=(".de", ".en"), fields=(german, english)
# #     )
# #
# #     # Creating vocabulary in each language
# #     german.build_vocab(train_data, max_size=10000, min_freq=2)
# #     english.build_vocab(train_data, max_size=10000, min_freq=2)
# #
# #     # Hyperparameters
# #     num_epochs = 1
# #     learning_rate = 0.001
# #     batch_size = 64
# #
# #     # Model hyperparameters
# #     load_model = False
# #     device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# #     input_size_encoder = len(german.vocab)
# #     input_size_decoder = len(english.vocab)
# #     output_size = len(english.vocab)
# #     encoder_embedding_size = 300
# #     decoder_embedding_size = 300
# #
# #     hidden_size = 1024
# #     num_layers = 2
# #     enc_dropout = 0.5
# #     dec_dropout = 0.5
# #
# #     train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
# #         (train_data, valid_data, test_data),
# #         batch_size=batch_size, sort_within_batch=True,
# #         sort_key=lambda x: len(x.src),
# #         device=device)
# #
# #     encoder_net = Encoder(input_size_encoder,
# #                           encoder_embedding_size,
# #                           hidden_size, num_layers).to(device)
# #
# #     decoder_net = Decoder(input_size_decoder,
# #                           decoder_embedding_size,
# #                           hidden_size, output_size, num_layers).to(device)
# #
# #     model = Seq2Seq(encoder_net, decoder_net).to(device)
# #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# #
# #     pad_idx = english.vocab.stoi['<pad>']
# #     criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# #
# #
# #     def translate_sentence(model, sentence, german, english, device, max_length=50):
# #         # print(sentence)
# #
# #         # sys.exit()
# #
# #         # Load german tokenizer
# #         spacy_ger = spacy.load("de_core_news_sm")
# #
# #         # Create tokens using spacy and everything in lower case (which is what our vocab is)
# #         if type(sentence) == str:
# #             tokens = [token.text.lower() for token in spacy_ger(sentence)]
# #         else:
# #             tokens = [token.lower() for token in sentence]
# #
# #         # print(tokens)
# #
# #         # sys.exit()
# #         # Add <SOS> and <EOS> in beginning and end respectively
# #         tokens.insert(0, german.init_token)
# #         tokens.append(german.eos_token)
# #
# #         # Go through each german token and convert to an index
# #         text_to_indices = [german.vocab.stoi[token] for token in tokens]
# #         # print(text_to_indices)
# #
# #         # Convert to Tensor
# #         sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
# #         # print('sentence tenssssor:::::', sentence_tensor)
# #
# #         # Build encoder hidden, cell state
# #         with torch.no_grad():
# #             hidden, cell = model.encoder(sentence_tensor)
# #
# #         outputs = [english.vocab.stoi["<sos>"]]
# #         # print('cccccccssssssssssssssssssssssssssssssssss', outputs)
# #
# #         for _ in range(max_length):
# #             previous_word = torch.LongTensor([outputs[-1]]).to(device)
# #             # print('ppppppppppppppppp', previous_word.size())
# #
# #             with torch.no_grad():
# #                 output, hidden, cell = model.decoder(previous_word, hidden, cell)
# #                 best_guess = output.argmax(1).item()
# #
# #             outputs.append(best_guess)
# #
# #             # Model predicts it's the end of the sentence
# #             if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
# #                 break
# #
# #         translated_sentence = [english.vocab.itos[idx] for idx in outputs]
# #
# #         # remove start token
# #         return translated_sentence[1:]
# #
# #
# #     def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
# #         print("=> Saving checkpoint")
# #         torch.save(state, filename)
# #
# #
# #     def load_checkpoint(checkpoint, model, optimizer):
# #         print("=> Loading checkpoint")
# #         model.load_state_dict(checkpoint["state_dict"])
# #         optimizer.load_state_dict(checkpoint["optimizer"])
# #
# #
# #     if load_model:
# #         load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
# #
# #     sentence = "Cristiano Ronaldo ist ein großartiger Fußballspieler mit erstaunlichen Fähigkeiten und Talenten."
# #
# #     for epoch in range(num_epochs):
# #         print(f"[Epoch {epoch} / {num_epochs}]")
# #
# #         checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
# #         save_checkpoint(checkpoint)
# #
# #         model.eval()
# #
# #         translated_sentence = translate_sentence(
# #             model, sentence, german, english, device, max_length=50
# #         )
# #
# #         # print(f"Translated example sentence: \n {translated_sentence}")
# #
# #         model.train()
# #
# #         for batch_idx, batch in enumerate(train_iterator):
# #             # Get input and targets and get to cuda
# #
# #             #
# #             # print(batch.src)
# #             # print(batch.trg)
# #
# #             inp_data = batch.src.to(device)
# #             target = batch.trg.to(device)
# #             # print('', inp_data)
# #             # print(inp_data.size())
# #             # Forward prop
# #             print(inp_data)
# #             output = model(inp_data, target)
# #
# #             # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
# #             # doesn't take input in that form. For example if we have MNIST we want to have
# #             # output to be: (N, 10) and targets just (N). Here we can view it in a similar
# #             # way that we have output_words * batch_size that we want to send in into
# #             # our cost fu nction, so we need to do some reshapin. While we're at it
# #             # Let's also remove the start token while we're at it
# #             output = output[1:].reshape(-1, output.shape[2])
# #             target = target[1:].reshape(-1)
# #
# #             optimizer.zero_grad()
# #             loss = criterion(output, target)
# #
# #             # Back prop
# #             loss.backward()
# #
# #             # Clip to avoid exploding gradient issues, makes sure grads are
# #             # within a healthy range
# #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
# #
# #             # Gradient descent step
# #             optimizer.step()
# #
# #             # # Plot to tensorboard
# #             # writer.add_scalar("Training loss", loss, global_step=step)
# #             # step += 1
# #
# # #
# # #
# # #
# # # class TalkingAgent(nn.Module):
# # #     def __init__(
# # #             self,
# # #             enc_embedding_size,
# # #             enc_hidden_size,
# # #             enc_num_layers,
# # #             dec_embedding_size,
# # #             dec_hidden_size,
# # #             dec_output_size,
# # #             dec_num_layers
# # #     ):
# # #         super().__init__()
# # #         self.Conv = nn.Sequential(
# # #             # 1st Conv layer
# # #             nn.Conv2d(1, 128, 3, stride=2),
# # #             nn.ReLU(),
# # #             # 2nd Conv layer
# # #             nn.Conv2d(128, 64, 3, stride=2),
# # #             nn.ReLU(),
# # #             nn.MaxPool2d(2, stride=2),
# # #             nn.BatchNorm2d(64, momentum=0.01),
# # #             nn.Dropout(0.2),
# # #             # 3rd Conv layer
# # #             nn.Conv2d(64, 32, 3, stride=2),
# # #             nn.ReLU(),
# # #             # 4th Conv layer
# # #             nn.Conv2d(32, 16, 3, stride=2),
# # #             nn.ReLU(),
# # #             nn.MaxPool2d(2, stride=2),
# # #             nn.Dropout(0.2)
# # #         )
# # #         # language module
# # #         self.MeaningEncoder = Encoder(
# # #             embedding_size=enc_embedding_size,
# # #             hidden_size=enc_hidden_size,
# # #             num_layers=enc_num_layers
# # #         )
# # #         self.MeaningDecoder = Decoder(
# # #             embedding_size=dec_embedding_size,
# # #             hidden_size=dec_hidden_size,
# # #             output_size=dec_output_size,
# # #             num_layers=dec_num_layers
# # #         )
# # #         self.LangModule = Seq2Seq(
# # #             encoder=self.MeaningEncoder,
# # #             decoder=self.MeaningDecoder,
# # #             target_len=decoder_embedding_size,
# # #             n_of_phonemes=dec_output_size
# # #         )
# # #         # cnn output (visual module output) + seq2seq output
# # #         self.FC1 = nn.Sequential(
# # #             nn.Linear(16 * 3 * 3 + 28, 100),
# # #             nn.ReLU()
# # #         )
# # #         self.FC2 = nn.Sequential(
# # #             nn.Linear(100, 40),
# # #             nn.ReLU()
# # #         )
# # #         self.FC3 = nn.Linear(40, 1)
# # #
# # #         self.out = nn.Sigmoid()
# # #
# # #     def forward(self, zero_frame, motion_frame, aux_sen=None):
# # #         # pass the frame zero thru conv layer
# # #         x_1 = self.Conv(zero_frame)  # size 32x16x3x3
# # #         x_1 = x_1.view(-1, x_1.size(1) * x_1.size(2) * x_1.size(3))  # size 32x144
# # #
# # #         # pass the motion field thru conv layer
# # #         x_2 = self.Conv(motion_frame)  # size 32x16x3x3
# # #         x_2 = x_2.view(-1, x_2.size(1) * x_2.size(2) * x_2.size(3))  # size 32x144
# # #
# # #         # concatenate ConvFixed and ConvMotion outputs
# # #         visual_vec = torch.cat((x_1, x_2), dim=1)  # size 32x288
# # #         #
# # #         # print(x_1[0])
# # #         # print(x_2[0])
# # #
# # #         # agent is a speaker
# # #         if aux_sen is None:
# # #             visual_seq = torch.stack((x_1, x_2), dim=0).permute([1, 0, 2])  # size
# # #             # print(visual_vec.size())
# # #
# # #             output_sentence = self.LangModule(visual_seq)
# # #         # agent is a listener
# # #         else:
# # #             output_sentence = aux_sen
# # #
# # #         return output_sentence
# # #
# # #
# # # class Encoder(nn.Module):
# # #     def __init__(self, embedding_size, hidden_size, num_layers):
# # #         super(Encoder, self).__init__()
# # #         self.dropout = nn.Dropout()
# # #         self.hidden_size = hidden_size
# # #         self.num_layers = num_layers
# # #
# # #         # self.embedding = nn.Embedding(input_size, embedding_size)
# # #
# # #         self.rnn = nn.LSTM(
# # #             embedding_size,
# # #             hidden_size,
# # #             num_layers,
# # #             dropout=0.2
# # #         )
# # #
# # #     def forward(self, x):
# # #         # x shape: (seq_length, N) where N is batch size
# # #         # embedding = self.dropout(self.embedding(x))
# # #         x = self.dropout(x)
# # #         # embedding shape: (seq_length, N, embedding_size)
# # #         _, (hidden, cell) = self.rnn(x)
# # #         # outputs shape: (seq_length, N, hidden_size)
# # #
# # #         return hidden, cell
# # #
# # #
# # # class Decoder(nn.Module):
# # #     def __init__(self, embedding_size, hidden_size, output_size, num_layers):
# # #         super(Decoder, self).__init__()
# # #         self.dropout = nn.Dropout(0.2)
# # #         self.hidden_size = hidden_size
# # #         self.num_layers = num_layers
# # #
# # #         # self.embedding = nn.Embedding(input_size, embedding_size)
# # #         self.rnn = nn.LSTM(
# # #             embedding_size,
# # #             hidden_size,
# # #             num_layers,
# # #             dropout=0.2
# # #         )
# # #         self.fc = nn.Linear(
# # #             hidden_size,
# # #             output_size
# # #         )
# # #
# # #     def forward(self, x, hidden, cell):
# # #         # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
# # #         # is 1 here because we are sending in a single word and not a sentence
# # #         x = x.unsqueeze(0)
# # #
# # #         x = self.dropout(x)
# # #         # embedding shape: (1, N, embedding_size)
# # #         outputs, (hidden, cell) = self.rnn(x, (hidden, cell))
# # #         # outputs shape: (1, N, hidden_size)
# # #
# # #         predictions = self.fc(outputs)
# # #
# # #         # predictions shape: (1, N, length_target_vocabulary) to send it to
# # #         # loss function we want it to be (N, length_target_vocabulary) so we're
# # #         # just gonna remove the first dim
# # #         predictions = predictions.squeeze(0)
# # #
# # #         return predictions, hidden, cell
# # #
# # #
# # # class Seq2Seq(nn.Module):
# # #     def __init__(self, encoder, decoder, target_len, n_of_phonemes):
# # #         super(Seq2Seq, self).__init__()
# # #         self.encoder = encoder
# # #         self.decoder = decoder
# # #         self.target_len = target_len  # 28
# # #         self.n_of_phonemes = n_of_phonemes  # 12
# # #
# # #     def forward(self, vis_seq):
# # #         outputs = torch.zeros(self.target_len, 1, self.n_of_phonemes)
# # #         # print('ssssssssssssssss', vis_seq.size(), vis_seq[0])
# # #         hidden, cell = self.encoder(vis_seq)
# # #         # Grab the first input to the Decoder which will be <SOS> token
# # #         x = outputs[0]
# # #         print('yyyyyyyyyy',hidden.size())
# # #         print('xxxxxxxxxxx', cell.size())
# # #
# # #         for t in range(1, self.target_len):  # 28
# # #             # Use previous hidden, cell as context from encoder at start
# # #             output, hidden, cell = self.decoder(x, hidden, cell)
# # #             # Store next output prediction
# # #
# # #             print('bbbbbbb',output.size())
# # #
# # #             # outputs[t] = output
# # #
# # #
# # #
# # #
# # #
# # #             # Get the best word the Decoder predicted (index in the vocabulary)
# # #             x = output.argmax(1)
# # #             print('kkkkkkkkk', x.item())
# # #
# # #         return outputs
