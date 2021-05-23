import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torch.nn.functional import log_softmax, softmax


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_size):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_size = embedding_size
        self.num_layers = 1
        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.embedding_size,
            num_layers=self.num_layers,
            dropout=0.2
        )

    def forward(self, x):
        # print('encoder input size:', x.size())  # 2 x batch_size x 144
        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(1), self.embedding_size).to(device)
                       )
        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(1), self.embedding_size).to(device)
                       )
        _, (hidden, cell) = self.rnn(x, (h_1, c_1))

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, embedding_size, n_features):
        super().__init__()
        self.n_features = n_features
        self.embedding_size = embedding_size

        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.embedding_size,
            num_layers=1,
            dropout=0.2
        )

        self.out = nn.Linear(self.embedding_size, self.n_features)

    def forward(self, x, hidden, cell):
        # print('decoder input size:', x.size())  # batch_size x 12
        x = x.reshape((1, x.size(1), self.n_features))  # 1 x batch_size x 12
        # print('decoder input after reshape', x.size())
        # print('hidden size', hidden.size())  # 2 x batch_size x 64
        # print('cell size', cell.size())  # 2 x batch_size x 64
        x, (hidden_n, cell_n) = self.rnn1(x, (hidden, cell))

        x = self.out(x)

        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):
    def __init__(
            self,
            seq_len,
            n_features_in,
            n_features_out,
            embedding_dim,
            output_length
    ):
        super().__init__()
        self.output_length = output_length
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.encoder = Encoder(seq_len, n_features_in, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, n_features_out).to(device)

    def forward(self, x):
        outputs = torch.zeros(self.output_length, x.size(1), self.n_features_out).to(device)  # TL x 1 x BS
        hidden, cell = self.encoder(x)
        # dummy input to start the sequence with
        x = torch.zeros((1, x.size(1), self.n_features_out)).to(device)  # 1 x BS x FO
        # mask = torch.ones((1, self.batch_size), dtype=torch.int64).to(device)

        for i in range(self.output_length):
            x, hidden, cell = self.decoder(x, hidden, cell)  # 1 x BS x FO

            # x_max = x.argmax(2)  # get the max index in each mini-batch --> 1 x BS
            # mask = torch.where(x_max == 0, 0, 1) * mask  # set value 0 as a padding token
            # a = x_max * mask
            # one-hot encoding -- each element becomes a FO size with the max element as 1 and else 0
            # x = torch.zeros(x.shape, device=device).scatter(2, a.unsqueeze(2), 1.0)
            # add to the outputs
            outputs[i] = x

        outputs = outputs.permute([1, 2, 0])  # BS x 1 x TL

        # return outputs.squeeze(1)  # BS x TL
        return outputs


def get_sentence_lengths(comm_input, vocab_size):
    return torch.sum(-(comm_input - vocab_size).sign(), dim=1)


def get_relevant_state(states, sentence_lengths):
    return torch.gather(
        states,
        dim=1,
        index=(sentence_lengths - 1).view(-1, 1, 1).expand(states.size(0), 1, states.size(2))
    )


class TalkingAgent(nn.Module):
    def __init__(self, event_vector_size=12, sen_emb_size=32, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size
        # self.Vision = nn.Sequential(
        #     # 1st Conv layer
        #     nn.Conv2d(1, 128, 3, stride=2),
        #     nn.ReLU(),
        #     # 2nd Conv layer
        #     nn.Conv2d(128, 64, 3, stride=2),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2, stride=2),
        #     # nn.BatchNorm2d(64, momentum=0.01),
        #     # nn.Dropout(0.2),
        #     # 3rd Conv layer
        #     nn.Conv2d(64, 32, 3, stride=2),
        #     nn.ReLU(),
        #     # 4th Conv layer
        #     nn.Conv2d(32, 16, 3, stride=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     # nn.Dropout(0.2)
        # )
        self.Vision = nn.Sequential(
            nn.Conv2d(1, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(180, 50),
            nn.ReLU(),
        )
        self.EventEmbedder = Seq2Seq(
            seq_len=2,
            n_features_in=50,
            n_features_out=15,
            embedding_dim=12,
            output_length=event_vector_size
        )
        # lang module
        self.embeddings = nn.Embedding(vocab_size + 2, sen_emb_size, padding_idx=vocab_size)
        self.gru = nn.GRU(input_size=sen_emb_size, hidden_size=sen_emb_size, batch_first=True)
        # reasoning module
        self.FC = nn.Sequential(
            nn.Linear(44, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, zero_frame, motion_frame, heard_sen):
        # seeing the event
        # pass the frame zero thru conv layer
        x_1 = self.Vision(zero_frame).reshape(zero_frame.size(0), -1)  # size BSx16x3x3
        x_1 = self.lin(x_1)
        # x_1 = x_1.reshape(-1, x_1.size(1) * x_1.size(2) * x_1.size(3))  # size BS x 144
        # pass the motion field thru conv layer
        x_2 = self.Vision(motion_frame).reshape(motion_frame.size(0), -1)  # size BSx16x3x3
        x_2 = self.lin(x_2)
        # x_2 = x_2.reshape(-1, x_2.size(1) * x_2.size(2) * x_2.size(3))  # size BS x 144
        visual_seq = torch.stack((x_1, x_2), dim=0)  # 2 x BS x 144
        event_emb = self.EventEmbedder(visual_seq)  # BS x TL
        # sentence generation
        sentence_lengths = get_sentence_lengths(heard_sen, vocab_size=self.vocab_size)
        output_sen = self.embeddings(heard_sen)
        output_sen, hidden = self.gru(output_sen)
        output_sen = get_relevant_state(output_sen, sentence_lengths).squeeze(1)

        # reasoning
        state_vec = torch.cat((event_emb.argmax(1), output_sen), dim=-1).float()
        pred = self.FC(state_vec)

        return log_softmax(pred, dim=1), softmax(pred, dim=1)[:, 1]


class TSAgent(nn.Module):
    def __init__(self, sen_emb_size=32, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.SpatialVision = nn.Sequential(
            nn.Conv2d(1, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        self.TemporalVision = nn.Sequential(
            nn.Conv2d(1, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        # lang module
        self.embeddings = nn.Embedding(vocab_size + 2, sen_emb_size, padding_idx=vocab_size)
        self.gru = nn.GRU(input_size=sen_emb_size, hidden_size=sen_emb_size, batch_first=True)
        # reasoning module
        self.FC = nn.Sequential(
            nn.Linear(180 + 180 + sen_emb_size, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, zero_frame, motion_frame, heard_sen):
        # seeing the event
        # pass the frame zero thru conv layer
        x_1 = self.SpatialVision(zero_frame).reshape(zero_frame.size(0), -1)  # size BSx16x3x3
        # x_1 = x_1.reshape(-1, x_1.size(1) * x_1.size(2) * x_1.size(3))  # size BS x 144
        # pass the motion field thru conv layer
        x_2 = self.TemporalVision(motion_frame).reshape(motion_frame.size(0), -1)  # size BSx16x3x3
        # x_2 = x_2.reshape(-1, x_2.size(1) * x_2.size(2) * x_2.size(3))  # size BS x 144

        # sentence generation
        sentence_lengths = get_sentence_lengths(heard_sen, vocab_size=self.vocab_size)
        output_sen = self.embeddings(heard_sen)
        output_sen, hidden = self.gru(output_sen)
        output_sen = get_relevant_state(output_sen, sentence_lengths).squeeze(1)

        # reasoning
        state_vec = torch.cat((output_sen, x_1, x_2), dim=-1).float()
        pred = self.FC(state_vec)

        return log_softmax(pred, dim=1), softmax(pred, dim=1)[:, 1]


class TSAgent2(nn.Module):
    def __init__(self, sen_emb_size=32, vocab_size=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.Vision = nn.Sequential(
            nn.Conv2d(1, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 20, 3, stride=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )
        # lang module
        self.embeddings = nn.Embedding(vocab_size + 2, sen_emb_size, padding_idx=vocab_size)
        self.gru = nn.GRU(input_size=sen_emb_size, hidden_size=sen_emb_size, batch_first=True)
        # reasoning module
        self.FC = nn.Sequential(
            nn.Linear(180 + 180 + sen_emb_size, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, zero_frame, motion_frame, heard_sen):
        # seeing the event
        # pass the frame zero thru conv layer
        x_1 = self.Vision(zero_frame).reshape(zero_frame.size(0), -1)  # size BSx16x3x3
        # x_1 = x_1.reshape(-1, x_1.size(1) * x_1.size(2) * x_1.size(3))  # size BS x 144
        # pass the motion field thru conv layer
        x_2 = self.Vision(motion_frame).reshape(motion_frame.size(0), -1)  # size BSx16x3x3
        # x_2 = x_2.reshape(-1, x_2.size(1) * x_2.size(2) * x_2.size(3))  # size BS x 144

        # sentence generation
        sentence_lengths = get_sentence_lengths(heard_sen, vocab_size=self.vocab_size)
        output_sen = self.embeddings(heard_sen)
        output_sen, hidden = self.gru(output_sen)
        output_sen = get_relevant_state(output_sen, sentence_lengths).squeeze(1)

        # reasoning
        state_vec = torch.cat((output_sen, x_1, x_2), dim=-1).float()
        pred = self.FC(state_vec)

        return log_softmax(pred, dim=1), softmax(pred, dim=1)[:, 1]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import OrganismsDataset

    training_set = OrganismsDataset()
    train_loader = DataLoader(training_set, batch_size=16, pin_memory=True, drop_last=True, shuffle=True)
    #
    speaker = TSAgent()
    speaker = speaker.to(device)
    #
    # listener = TalkingAgent()
    # listener = listener.to(device)
    #
    ds = iter(train_loader)
    #
    dummy_1 = next(ds)
    results, sentences = speaker(dummy_1[0].to(device), dummy_1[1].to(device), heard_sen=None)
    print('SPEAKER\n', sentences.argmax(1))
    #
    # dummy_2 = next(ds)
    # lis_results, lis_sentences = listener(dummy_2[0].to(device), dummy_2[1].to(device), heard_sen=sentences)
    # print('LISTENER\n', torch.round(lis_results))
    #
    # print([isinstance(i, LangModule) for i in speaker.children()])

    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 144)

    print(list(i[0] for i in speaker.named_parameters()))

    x, y = dummy_1[0].to(device).size(), dummy_1[1].to(device).size()

    # alexnet.to(device)
    # alexnet(dummy_1[0].to(device))
