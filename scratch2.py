import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings

warnings.filterwarnings('ignore')
from torch.autograd import Variable
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_size, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_size = embedding_size
        self.num_layers = 2
        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.embedding_size,
            num_layers=self.num_layers,
            dropout=0.2
        )

    def forward(self, x):
        # print('encoder input size:', x.size())  # 2 x batch_size x 144
        h_1 = Variable(torch.zeros(
            self.num_layers, self.batch_size, self.embedding_size).to(device)
                       )
        c_1 = Variable(torch.zeros(
            self.num_layers, self.batch_size, self.embedding_size).to(device)
                       )
        _, (hidden, cell) = self.rnn(x, (h_1, c_1))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, embedding_size, n_features, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.n_features = n_features
        self.embedding_size = embedding_size

        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.embedding_size,
            num_layers=2,
            dropout=0.2
        )

        self.out = nn.Linear(self.embedding_size, self.n_features)

    def forward(self, x, hidden, cell):
        # print('decoder input size:', x.size())  # batch_size x 12
        x = x.reshape((1, self.batch_size, self.n_features))  # 1 x batch_size x 12
        # print('decoder input after reshape', x.size())
        # print('hidden size', hidden.size())  # 2 x batch_size x 64
        # print('cell size', cell.size())  # 2 x batch_size x 64
        x, (hidden_n, cell_n) = self.rnn1(x, (hidden, cell))

        x = self.out(x)

        return x, hidden_n, cell_n


class LangUtterance(nn.Module):
    def __init__(
            self,
            seq_len,
            n_features_in,
            n_features_out,
            embedding_dim,
            output_length,
            batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.output_length = output_length
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.encoder = Encoder(seq_len, n_features_in, embedding_dim, batch_size).to(device)
        self.decoder = Decoder(embedding_dim, n_features_out, batch_size).to(device)

    def forward(self, x):
        outputs = torch.zeros(self.output_length, 1, self.batch_size).to(device)  # TL x 1 x BS
        hidden, cell = self.encoder(x)
        # dummy input to start the sequence with
        x = torch.zeros((1, self.batch_size, self.n_features_out)).to(device)  # 1 x BS x FO
        mask = torch.ones((1, self.batch_size), dtype=torch.int64).to(device)

        for i in range(self.output_length):
            x, hidden, cell = self.decoder(x, hidden, cell)  # 1 x BS x FO

            x_max = x.argmax(2)  # get the max index in each mini-batch --> 1 x BS
            mask = torch.where(x_max == 0, 0, 1) * mask  # set value 0 as a padding token
            a = x_max * mask
            # one-hot encoding -- each element becomes a FO size with the max element as 1 and else 0
            x = torch.zeros(x.shape, device=device).scatter(2, a.unsqueeze(2), 1.0)
            # add to the outputs
            outputs[i] = a

        outputs = outputs.permute([2, 1, 0])  # BS x 1 x TL

        return outputs.squeeze(1)  # BS x TL


class TalkingAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv = nn.Sequential(
            # 1st Conv layer
            nn.Conv2d(1, 128, 3, stride=2),
            nn.ReLU(),
            # 2nd Conv layer
            nn.Conv2d(128, 64, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.Dropout(0.2),
            # 3rd Conv layer
            nn.Conv2d(64, 32, 3, stride=2),
            nn.ReLU(),
            # 4th Conv layer
            nn.Conv2d(32, 16, 3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.2)
        )
        # language module
        self.LangModule = LangUtterance(
            seq_len=2,
            n_features_in=16 * 3 * 3,
            n_features_out=14,
            embedding_dim=64,
            output_length=12,
            batch_size=16
        )
        # cnn output (visual module output) + seq2seq output to be fed here
        self.FC1 = nn.Sequential(
            nn.Linear(16 * 3 * 3 * 2 + self.LangModule.output_length, 100),
            nn.ReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(100, 40),
            nn.ReLU()
        )
        self.FC3 = nn.Linear(40, 1)
        self.OUT = nn.Sigmoid()

    def forward(self, zero_frame, motion_frame, aux_sen=None):
        # pass the frame zero thru conv layer
        x_1 = self.Conv(zero_frame)  # size BSx16x3x3
        x_1 = x_1.view(-1, x_1.size(1) * x_1.size(2) * x_1.size(3))  # size BS x 144

        # pass the motion field thru conv layer
        x_2 = self.Conv(motion_frame)  # size BSx16x3x3
        x_2 = x_2.view(-1, x_2.size(1) * x_2.size(2) * x_2.size(3))  # size BS x 144

        # agent is a speaker
        if aux_sen is None:
            visual_seq = torch.stack((x_1, x_2), dim=0)  # 2 x BS x 144
            output_sentence = self.LangModule(visual_seq)  # BS x TL
            # concatenate ConvFixed and ConvMotion and sentence (from lang module) outputs
            visual_vec = torch.cat((x_1, x_2, output_sentence), dim=1)  # size BS x 288
        # agent is a listener
        else:
            # concatenate ConvFixed and ConvMotion  outputs with speaker's sentence
            visual_vec = torch.cat((x_1, x_2, aux_sen), dim=1)  # size 32x288
            output_sentence = None  # listener doesn't speak out

        state_vec = self.FC1(visual_vec)
        state_vec = self.FC2(state_vec)
        state_vec = self.FC3(state_vec)
        final_result = self.OUT(state_vec)

        return final_result, output_sentence


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import OrganismsDataset

    training_set = OrganismsDataset()
    train_loader = DataLoader(training_set, batch_size=16, pin_memory=True, drop_last=True, shuffle=True)

    speaker = TalkingAgent()
    speaker = speaker.to(device)

    listener = TalkingAgent()
    listener = listener.to(device)

    ds = iter(train_loader)

    dummy_1 = next(ds)
    results, sentences = speaker(dummy_1[0].to(device), dummy_1[1].to(device))
    print('SPEAKER\n', sentences)

    dummy_2 = next(ds)
    lis_results, lis_sentences = listener(dummy_2[0].to(device), dummy_2[1].to(device), aux_sen=sentences)
    print('LISTENER\n', lis_results)
