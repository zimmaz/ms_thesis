import numpy as np
import torch


def decode(model, still_frames, motion_frames, max_sentence_len, vocab_size, device):
    '''
        Given the model and one batch images, greedy generate the most probable message.
        Input:
            model: the agent
            all_inputs: one batch of pictures (one in the pair)
        Output:
            actions: (batch_size, max_sent_len), the generated message
            all_probs: (batch_size,), the success prob. of message describing the image.
    '''
    zero_frames = still_frames.permute(0, 2, 3, 1)
    motion_fileds = motion_frames.permute(0, 2, 3, 1)
    relevant_procs = list(range(zero_frames.size(0)))  # Indicator in one batch

    # Initial the output sentence and prob with -1, for each figure in one batch
    actions = np.array([[-1 for _ in range(max_sentence_len)] for _ in relevant_procs])  # output sentence
    all_probs = np.array([-1. for _ in relevant_procs])

    for l in range(max_sentence_len):
        still_frames = zero_frames[relevant_procs]
        motion_frames = motion_fileds[relevant_procs]

        batch_size = still_frames.size(0)
        # 50*5, represent the symbol for each figure in one batch
        next_symbol = np.tile(np.expand_dims(np.arange(0, vocab_size), 1), batch_size).transpose()

        # Here run_communications stores the existing best sequence, and also the fresh one waiting to choose.
        if l > 0:
            run_communications = np.concatenate((np.expand_dims(actions[relevant_procs, :l].transpose(),
                                                                2).repeat(vocab_size, axis=2),
                                                 np.expand_dims(next_symbol, 0)), axis=0)
        else:
            run_communications = np.expand_dims(next_symbol, 0)

        # Expand inputs to 5*50, i.e. each vocab have one img, then feed 250 imgs and texts to agents
        # to get 250*1 probabilities, then reshape it to 50*5, each row represent the probability of
        # choosing specific vocab to communicate. We then select the best one in vocabulary, and use
        # sel_comm_idx to record the chosen result.
        expanded_inputs_still = still_frames.repeat(vocab_size, 1, 1, 1)
        # print('hahahahah', expanded_inputs_still.size())
        expanded_inputs_motion = motion_frames.repeat(vocab_size, 1, 1, 1)

        logits, probs = model(expanded_inputs_still.permute(0, 3, 1, 2),
                              expanded_inputs_motion.permute(0, 3, 1, 2),
                              torch.Tensor(run_communications.transpose().reshape(-1, 1 + l)).long().to(device))
        probs = probs.view((vocab_size, batch_size)).transpose(0, 1)

        probs, sel_comm_idx = torch.max(probs, dim=1)

        comm = run_communications[:, np.arange(len(relevant_procs)), sel_comm_idx.data.cpu().numpy()].transpose()

        # If any img can achieve prob>0.95, it is finished and we remove it from relevant_procs,
        # and store the actions.
        finished_p = []
        for i, (action, p, prob) in enumerate(zip(comm, relevant_procs, probs)):
            if prob > 0.95:
                finished_p.append(p)
                if prob.item() < 0:
                    continue
            # Store the converged actions.
            for j, symb in enumerate(action):
                actions[p][j] = symb

            all_probs[p] = prob

        for p in finished_p:
            relevant_procs.remove(p)

        if len(relevant_procs) == 0:
            break

    actions[actions == -1] = vocab_size  # padding token
    actions = torch.Tensor(np.array(actions)).long().to(device)

    return actions, all_probs