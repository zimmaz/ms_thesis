import os
import random
import warnings
import gc

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from visdom import Visdom

import obverter
from agents import TalkingAgent, TSAgent, TSAgent2
from data import OrganismsDataset
from utils import *

seed = 365
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

warnings.filterwarnings('ignore')

gc.collect()
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 24
vocab_size = 5
max_sen_len = 10


# instantiate the agents
agent_a = TSAgent2(vocab_size=vocab_size).to(device)
agent_b = TSAgent2(vocab_size=vocab_size).to(device)

# loss function
# criterion_sp = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
criterion = nn.NLLLoss()

# optimizers
optimizer_a_gen = optim.Adam([p for p in agent_a.parameters() if p.requires_grad], lr=6e-4)
optimizer_b_gen = optim.Adam([p for p in agent_b.parameters() if p.requires_grad], lr=6e-4)

# load data and split it to train and test sets
training_set = OrganismsDataset(path='organisms/', return_pair=True, n_pairs=1000)
test_set = OrganismsDataset('./organisms/test/', return_pair=True, n_pairs=100)
test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True, drop_last=True)

# setup dashboard for monitoring the training process
# viz = Visdom()
# viz.line([0.], [0], win='agent A', opts=dict(title='loss (Agent A)'))
# viz.line([0.], [0], win='agent B', opts=dict(title='loss (Agent B)'))

# randomly sample organisms from each category from the test dataset
# samples = test_set.pick_samples_from_each_cat(3)


def train_round(speaker, listener, optimizer, max_sentence_len, vocab_size):
    speaker.train(False)
    listener.train(True)
    # speaker.Vision.requires_grad_(False)
    # listener.Vision.requires_grad_(False)

    round_total = 0
    round_correct = 0
    round_loss = 0
    round_sentence_length = 0

    # pick new pairs in each round
    training_set.refresh_pairs()
    train_loader = DataLoader(training_set, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)

    for i, data in enumerate(train_loader):
        # load a pair of organisms from training set
        still_frames_1 = data[0][0].float().to(device)
        motion_frames_1 = data[0][1].float().to(device)
        still_frames_2 = data[1][0].float().to(device)
        motion_frames_2 = data[1][1].float().to(device)
        speaker_obj = data[0][2], data[0][3]
        listener_obj = data[1][2], data[1][3]

        # label indicates whether the same category of organisms is shown
        labels = torch.tensor(data[2], device=device).float()
        # Play the game and backpropagation
        speaker_actions, speaker_probs = obverter.decode(
            speaker, still_frames_1, motion_frames_1, max_sentence_len,
                                                         vocab_size, device)

        lg, probs = listener(still_frames_2, motion_frames_2, speaker_actions)
        predictions = torch.round(probs).long()  # Convert the probabilities to 0/1 predictions
        correct_vector = (predictions == labels).float()
        n_correct = correct_vector.sum()  # out of batch_size
        listener_loss = criterion(lg, labels.long())

        optimizer.zero_grad()
        listener_loss.backward()
        optimizer.step()

        for t in zip(
                speaker_actions,
                speaker_probs,
                list(zip(speaker_obj[0], speaker_obj[1])),
                list(zip(listener_obj[0], listener_obj[1])),
                labels,
                probs
        ):
            speaker_action, speaker_prob, speaker_obj, listener_obj, label, listener_prob = t
            message = get_message(speaker_action, vocab_size)
            print("message: '%s', speaker object: %s, speaker score: %.2f, listener object: %s, label: %d, ""listener score: %.2f"
                  % (message, speaker_obj, speaker_prob, listener_obj, label.item(), listener_prob.item()))

        print("batch accuracy", n_correct.item() / len(still_frames_1))
        print("batch loss", listener_loss.item())

        # ========== Stats for a round (e.g. 20 games, each game have 50 batch_size data)
        round_correct += n_correct
        round_total += len(still_frames_1)
        round_loss += listener_loss * len(still_frames_1)  # !!!! Why * len(input1)???
        round_sentence_length += (speaker_actions < vocab_size).sum(dim=1).float().mean() * len(still_frames_1)

    round_accuracy = (round_correct / round_total).item()
    round_loss = (round_loss / round_total).item()
    round_sentence_length = (round_sentence_length / round_total).item()

    return round_accuracy, round_loss, round_sentence_length


if __name__ == '__main__':
    agent1_accuracy_history = []
    agent1_message_length_history = []
    agent1_loss_history = []

    os.makedirs('checkpoints', exist_ok=True)

    for r in range(3000):
        print("********** round %d **********" % r)
        round_accuracy, round_loss, round_sentence_length = train_round(
            agent_a,
            agent_b,
            optimizer_b_gen,
            max_sen_len,
            vocab_size
        )
        print_round_stats(round_accuracy, round_loss, round_sentence_length)

        agent1_accuracy_history.append(round_accuracy)
        agent1_message_length_history.append(round_sentence_length / 20)
        agent1_loss_history.append(round_loss)

        r += 1
        print("replacing roles")
        print("********** round %d **********" % r)

        round_accuracy, round_loss, round_sentence_length = train_round(
            agent_b,
            agent_a,
            optimizer_a_gen,
            max_sen_len,
            vocab_size
        )
        print_round_stats(round_accuracy, round_loss, round_sentence_length)

        t = list(range(len(agent1_accuracy_history)))
        plt.plot(t, agent1_accuracy_history, label="Accuracy")
        plt.plot(t, agent1_message_length_history, label="Message length (/10)")
        plt.plot(t, agent1_loss_history, label="Training loss")

        plt.xlabel('# Rounds')
        plt.legend()
        plt.savefig("graph.png")
        plt.clf()

        if r % 50 == 0:
            torch.save(agent_a.state_dict(), os.path.join('checkpoints', 'agent1-%d.ckp' % r))
            torch.save(agent_b.state_dict(), os.path.join('checkpoints', 'agent2-%d.ckp' % r))

#
# def test(agent_a_, agent_b_, test_set_, samples_, round_):
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
#                 print(frame_zero.size())
#                 row = {
#                     'org_path': o,
#                     'round': round_,
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
#         # lan_loss_a = 0.0
#         # lan_loss_b = 0.0
#         print(f'############ round: {epoch} ############')
#         for i, data in enumerate(train_loader):
#             batch_n = epoch * len(train_loader) + i + 1
#             # load a pair of organisms from training set
#             frame_sp = data[0][0]
#             motion_sp = data[0][1]
#             frame_ls = data[1][0]
#             motion_ls = data[1][1]
#             # label indicates whether the same category of organisms is shown
#             label = torch.tensor(data[2], device=device).to(torch.float32).unsqueeze(1)
#             # pick the speaker randomly
#             if random.random() > 0.5:
#                 assigned_agent = 'a'
#                 speaker = agent_a
#                 listener = agent_b
#                 # optimizer_sp = optimizer_a_lan
#                 optimizer = optimizer_b_gen
#             else:
#                 assigned_agent = 'b'
#                 speaker = agent_b
#                 listener = agent_a
#                 # optimizer_sp = optimizer_b_lan
#                 optimizer = optimizer_a_gen
#
#             # set both optimizers params to zero
#             optimizer.zero_grad()
#             # optimizer_ls.zero_grad()
#
#             # only update speaker's language module
#             # listener.Conv.requires_grad_(False)
#             # listener.LangModule.requires_grad_(False)
#             # speaker.FC.requires_grad_(False)
#             # speaker.Conv.requires_grad_(False)
#
#             # what speaker says about the organism shown to it
#             with torch.no_grad():
#                 _, sentence_sp = speaker(frame_sp.to(device), motion_sp.to(device))
#
#             # give the speaker's guess to the listener and let it  guess if it sees the same organism as speaker's
#             listener_call, sentence_ls = listener(frame_ls.to(device), motion_ls.to(device), heard_sen=sentence_sp)
#             loss = criterion(listener_call, label)
#             loss.backward()
#             optimizer.step()
#
#             # # listener speaks out to train the speaker
#             # # _, sentence_ls = listener(frame_ls.to(device), motion_ls.to(device))
#             # #
#             # loss_sp = criterion_sp(sentence_sp, sentence_ls.argmax(1).to(torch.long))
#             # loss_sp.backward()
#             # #
#             # # optimizer_sp.step()
#             #
#             if not assigned_agent == 'a':
#                 running_loss_a += loss.item()
#             else:
#                 running_loss_b += loss.item()
#             if batch_n % 10 == 9:
#                 print(f'______{batch_n}______\n')
#                 print('Speaker sentence \n', sentence_sp.argmax(1))
#                 print('Listener sentence \n', sentence_ls.argmax(1))
#
#                 # if assigned_agent == 'b':
#
#                 # lan_loss_b += loss_sp.item()
#                 viz.line([running_loss_a / 10], [batch_n], win='agent A', update='append', name='lossA')
#                 viz.line([running_loss_b / 10], [batch_n], win='agent B', update='append', name='lossB')
#                 # else:
#                 #     running_loss_a += loss_ls.item()
#                 #     # lan_loss_a += loss_sp.item()
#                 #     # viz.line([lan_loss_a/10], [batch_n], win='agent A', update='append', name='lang loss')
#                 #     viz.line([running_loss_a / 10], [batch_n], win='agent A', update='append', name='gen loss')
#
#                 running_loss_a = 0.0
#                 running_loss_b = 0.0
            #     # lan_loss_a = 0.0
            #     # lan_loss_b = 0.0
            #
            # if epoch % 50 == 0:
            #     torch.save(agent_a.state_dict(), os.path.join('checkpoints', f'test-agentAV2-{epoch + 1}.ckp'))
            #     torch.save(agent_b.state_dict(), os.path.join('checkpoints', f'test-agentBV2-{epoch + 1}.ckp'))

            # for child in speaker.children():
            #     for p in child.parameters():
            #         p.requires_grad = True

            # if batch_n % 10 == 9:
            #     if assigned_agent == 'a':
            #         running_loss_a += loss_sp.item()
            #         # viz.line([loss_sp.item()], [batch_n], win='agent A', update='append')
            #         viz.line([running_loss_a/10], [batch_n], win='agent A', update='append')
            #     else:
            #         running_loss_b += loss_sp.item()
            #         # viz.line([loss_sp.item()], [batch_n], win='agent B', update='append')
            #         viz.line([running_loss_b / 10], [batch_n], win='agent B', update='append')

            # with torch.no_grad():
            # _, _, sentence_sp = speaker(frame_sp.to(device), motion_sp.to(device))
            # sentence_sp, _ = speaker(frame_sp.to(device), motion_sp.to(device))
            # sentence_sp.requires_grad = True
            # print(f'++++++agent {assigned_agent}+++++\n\n', sentence_sp)

            # listener training
            # vis_emb, lin_emb, _ = listener(frame_ls.to(device), motion_ls.to(device), aux_sen=sentence_sp)

            # turn off grad in language module only in listener model
            # for p in listener.LangModule.parameters():
            #     p.requires_grad = False
            # for p in listener.Lang_out.parameters():
            #     p.requires_grad = False

            # loss_ls = criterion(vis_emb, lin_emb, torch.tensor(label, device=device))

            # for p in speaker.Conv.parameters():
            #     p.requires_grad = True
            # for p in listener.LangModule.parameters():
            #     p.requires_grad = True

            # for p in speaker.Vis_out.parameters():
            #     p.requires_grad = True
            # for p in speaker.Lang_out.parameters():
            #     p.requires_grad = True

            # if i % 20 == 19:  # print every 20 mini-batches
            #     print('[%d, %5d] agent A loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss_a / 20))
            #     print('[%d, %5d] agent B loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss_b / 20))
            #
            #     running_loss_a = 0.0
            #     running_loss_b = 0.0

        #     loss = loss_fn(output.to(device), torch.tensor(labels).to(device))
        #     loss.backward()
        #     optimizer.step()
        #
        #     # predictions = torch.round(probs)
        #     # correct_vector = [1 for p, q in zip(predictions.tolist(), labels) if p == q]
        #     # n_correct = sum(correct_vector)
        #     # print("batch accuracy", n_correct / len(frame_zero))
        #     # print("batch loss", loss.item())
        #
        #     batch_n = epoch * len(train_loader) + i + 1
        #
        #     viz.line([loss.item()], [batch_n], win='train_loss', update='append')
        #     # viz.line([n_correct / len(frame_zero)], [batch_n], win='accuracy', update='append')
        #
        # if epoch % 20 == 0:
        #     torch.save(silent_agent.state_dict(), os.path.join('checkpoints', f'test-multi-{epoch + 1}.ckp'))

# def evaluate():
#     data_test_iter = iter(test_loader)
#
#     images, mfs, kind, motion_type, orgs = next(data_test_iter)
#
#     # print images
#     # imshow(torchvision.utils.make_grid(images))
#     # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#     silent_agent.load_state_dict(torch.load(r'checkpoints/test-multi-21.ckp'))
#
#     silent_agent.eval()
#
#     for layer in silent_agent.children():
#         if hasattr(layer, '__len__'):
#             for s_layer in range(len(layer)):
#                 print(s_layer)
#                 if type(layer[s_layer]) == nn.BatchNorm2d:
#                     layer[s_layer].train()
#                     layer[s_layer].momentum = 0.01
#
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for data in test_loader:
#             images, mfs, kind, motion_type, orgs = data
#             labels = [categorize(k, m) for k, m in zip(kind, motion_type)]
#             labels = torch.tensor(labels).to(device)
#             outputs = silent_agent(images.to(device), mfs.to(device))
#
#             # log_, probs = log_softmax(outputs, dim=1), softmax(outputs, dim=1)[:, 1]
#
#             _, predictions = torch.max(outputs, 1)
#             # correct_vector = [1 for p, q in zip(predictions.tolist(), labels) if p == q]
#             total += labels.size(0)
#             correct += (predictions == labels).sum().item()
#             # n_correct = sum(correct_vector)
#             # print(f'{n_correct} out of {len(predictions)}')
#             # print("batch accuracy", n_correct / len(predictions))
#
#     print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}')
#
#
# if __name__ == '__main__':
#     pass
#     train()
#     # evaluate()
