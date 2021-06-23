import os
import warnings
import gc
import pickle
import random

import torch.optim as optim
from torch.utils.data import DataLoader
from visdom import Visdom

import obverter
from agents import TalkingAgentV2
from data import OrganismsDataset
from utils import *


warnings.filterwarnings('ignore')
gc.collect()
torch.cuda.empty_cache()

# initialize random number generator to make the process deterministic
seed = 365
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# run it on GPU if cuda available (recommended)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dashboard = False
batch_size = 20
vocab_size = 5
max_sen_len = 10

# instantiate the agents
agent_a = TalkingAgentV2(vocab_size=vocab_size, sen_emb_size=64).to(device)
agent_b = TalkingAgentV2(vocab_size=vocab_size, sen_emb_size=64).to(device)

# loss function
criterion = nn.NLLLoss()

# optimizers
optimizer_a_gen = optim.Adam([p for p in agent_a.parameters() if p.requires_grad], lr=1e-3)
optimizer_b_gen = optim.Adam([p for p in agent_b.parameters() if p.requires_grad], lr=1e-3)

# load training data
training_set = OrganismsDataset(
    path='organisms/',
    return_pair=True,
    n_pairs=1000,
    return_all_frames=True,
    n_organisms=2400
)

# setup dashboard for monitoring the training process
if dashboard:
    viz = Visdom()
    viz.line([0.], [0], win='agent A', opts=dict(title='Training Stats (Agent A)'))
    viz.line([0.], [0], win='agent B', opts=dict(title='Training Stats (Agent B)'))


def train_round(speaker, listener, optimizer, max_sentence_len, vocab_size, train_loader):
    speaker.train(False)
    listener.train(True)

    round_total = 0
    round_correct = 0
    round_loss = 0
    round_sentence_length = 0

    for i, data in enumerate(train_loader):
        # load a pair of organisms from training set
        frames_1 = data[0][0]
        frames_2 = data[1][0]
        speaker_obj = data[0][2], data[0][3], data[0][1]
        listener_obj = data[1][2], data[1][3], data[1][1]
        # label indicates whether the same category of organisms is shown to agents
        labels = torch.tensor(data[2], device=device).float()
        # let the speaker find the best sentence to describe its given event
        speaker_actions, speaker_probs = obverter.decode_video(speaker, frames_1, max_sentence_len, vocab_size, device)
        # test the listener
        lg, probs = listener(frames_2.permute(0, 2, 1), speaker_actions)
        predictions = torch.round(probs).long()  # Convert the probabilities to binary predictions
        correct_vector = (predictions == labels).float()
        n_correct = correct_vector.sum()
        listener_loss = criterion(lg, labels.long())

        optimizer.zero_grad()
        listener_loss.backward()
        optimizer.step()

        for t in zip(
                speaker_actions,
                speaker_probs,
                list(zip(speaker_obj[0], speaker_obj[1], speaker_obj[2])),
                list(zip(listener_obj[0], listener_obj[1], listener_obj[2])),
                labels,
                probs
        ):
            speaker_action, speaker_prob, speaker_obj, listener_obj, label, listener_prob = t
            message = get_message(speaker_action, vocab_size)
            print(f'message: "{message}", speaker object: {speaker_obj}, speaker score: {speaker_prob:.2f}, '
                  f'listener object: {listener_obj}, label: {label.item()}, listener score: {listener_prob.item():.2f}')

        print('batch accuracy', n_correct.item() / len(frames_1))
        print('batch loss', listener_loss.item())

        round_correct += n_correct
        round_total += len(frames_1)
        round_loss += listener_loss * len(frames_1)
        round_sentence_length += (speaker_actions < vocab_size).sum(dim=1).float().mean() * len(frames_1)

    round_accuracy = (round_correct / round_total).item()
    round_loss = (round_loss / round_total).item()
    round_sentence_length = (round_sentence_length / round_total).item()

    return round_accuracy, round_loss, round_sentence_length


if __name__ == '__main__':
    agent_a_accuracy_history = []
    agent_a_message_length_history = []
    agent_a_loss_history = []

    agent_b_accuracy_history = []
    agent_b_message_length_history = []
    agent_b_loss_history = []

    os.makedirs('checkpoints', exist_ok=True)

    for r in range(1, 6001):
        training_set.refresh_pairs()
        train_loader = DataLoader(training_set, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)

        print("********** round %d **********" % r)

        round_accuracy_a, round_loss_a, round_sentence_length_a = train_round(
            agent_b,
            agent_a,
            optimizer_a_gen,
            max_sen_len,
            vocab_size,
            train_loader
        )
        print_round_stats(round_accuracy_a, round_loss_a, round_sentence_length_a)

        agent_a_accuracy_history.append(round_accuracy_a)
        agent_a_message_length_history.append(round_sentence_length_a / max_sen_len)
        agent_a_loss_history.append(round_loss_a)

        if dashboard:
            viz.line([round_accuracy_a], [r], win='agent A', update='append', name='Acc')
            viz.line([round_loss_a], [r], win='agent A', update='append', name='Loss')
            viz.line(
                [round_sentence_length_a / max_sen_len],
                [r],
                win='agent A',
                update='append',
                name=f'Message Length / {max_sen_len}'
            )

        print('replacing roles')
        print(f'********** round {r} **********')

        round_accuracy_b, round_loss_b, round_sentence_length_b = train_round(
            agent_a,
            agent_b,
            optimizer_b_gen,
            max_sen_len,
            vocab_size,
            train_loader
        )
        print_round_stats(round_accuracy_b, round_loss_b, round_sentence_length_b)

        agent_b_accuracy_history.append(round_accuracy_b)
        agent_b_message_length_history.append(round_sentence_length_b / max_sen_len)
        agent_b_loss_history.append(round_loss_b)

        if dashboard:
            viz.line([round_accuracy_b], [r], win='agent B', update='append', name='Acc')
            viz.line([round_loss_b], [r], win='agent B', update='append', name='Loss')
            viz.line(
                [round_sentence_length_b / max_sen_len],
                [r],
                win='agent B',
                update='append',
                name=f'Message Length / {max_sen_len}'
            )

        if r % 50 == 0:
            torch.save(agent_a.state_dict(), os.path.join('checkpoints', f'agent1-{r}.ckp'))
            torch.save(agent_b.state_dict(), os.path.join('checkpoints', f'agent2-{r}.ckp'))

            agent_a_stats = agent_a_accuracy_history, agent_a_loss_history, agent_a_message_length_history
            with open('agent_a_train_stats.pkl', 'wb') as f:
                pickle.dump(agent_a_stats, f)

            agent_b_stats = agent_b_accuracy_history, agent_b_loss_history, agent_b_message_length_history
            with open('agent_b_train_stats.pkl', 'wb') as f:
                pickle.dump(agent_b_stats, f)

        # t = list(range(len(agent_a_accuracy_history)))
        # plt.plot(t, agent_a_accuracy_history, label="Accuracy")
        # plt.plot(t, agent_a_message_length_history, label="Message length (/10)")
        # plt.plot(t, agent_a_loss_history, label="Training loss")
        #
        # plt.xlabel('# Rounds')
        # plt.legend()
        # plt.savefig("graph.png")
        # plt.clf()