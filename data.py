import glob
import os
import random
import re

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# params for ShiTomasi corner detection
feature_params = {
    'maxCorners': 100,
    'qualityLevel': 0.3,
    'minDistance': 7,
    'blockSize': 7
}

# Parameters for lucas kanade optical flow
lk_params = {
    'winSize': (10, 10),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}


def is_edible(kind, motion_type):
    if kind == 'edible':
        if motion_type in ('spin', 'still'):
            return 1
        else:
            return 0
    elif kind == 'poisonous':
        if motion_type in ('still', 'move'):
            return 0
        else:
            return 1


def categorize(kind, motion_type):
    if kind == 'edible':
        if motion_type == 'still':
            return 1
        elif motion_type == 'spin':
            return 2
        elif motion_type == 'move':
            return 3
    elif kind == 'poisonous':
        if motion_type == 'still':
            return 4
        elif motion_type == 'spin':
            return 5
        elif motion_type == 'move':
            return 6


class OrganismsDataset(Dataset):
    def __init__(self, path=r'./organisms/', return_pair=False, n_pairs=1000):
        self.path = path
        self.organisms = [org for org in glob.iglob(f'{path}/*.gif')]
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ]
        )
        self.categories = {cat: [] for cat in range(1, 7)}
        # divide organisms into 6 distinct categories
        for o in glob.iglob(f'{self.path}/*.gif'):
            # cat 1 (edible-still)
            if ('edible' in o) and ('still' in o):
                self.categories[1].append(o)
            # cat 2 (edible-spin)
            if ('edible' in o) and ('spin' in o):
                self.categories[2].append(o)
            # cat 3 (edible-move)
            if ('edible' in o) and ('move' in o):
                self.categories[3].append(o)
            # cat 4 (poisonous-still)
            if ('poisonous' in o) and ('still' in o):
                self.categories[4].append(o)
            # cat 5 (poisonous-spin)
            if ('poisonous' in o) and ('spin' in o):
                self.categories[5].append(o)
            # cat 6 (poisonous-move)
            if ('poisonous' in o) and ('move' in o):
                self.categories[6].append(o)

        self.return_pair = return_pair
        self.n_pairs = n_pairs

        if return_pair:
            self.pairs = self.set_pairs()

    def set_pairs(self):
        return [self.pick_a_pair() for _ in range(self.n_pairs)]

    def refresh_pairs(self):
        self.pairs = self.set_pairs()

    def preprocess(self, img_path):
        # get the first frame of the gif file
        cap = cv2.VideoCapture(img_path)
        ret, frame_zero = cap.read()

        motion_file = img_path.replace('.gif', '_of.png')
        motion_field = cv2.imread(motion_file)

        if not os.path.isfile(motion_file) or motion_field is None:
            old_gray = cv2.cvtColor(frame_zero, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            # Create a mask image for drawing purposes
            motion_field = np.zeros_like(frame_zero)

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                if p1 is None:
                    continue

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # draw the tracks
                for (new, old) in zip(good_new, good_old):
                    a, b = new.ravel().astype(np.uint16)
                    c, d = old.ravel().astype(np.uint16)
                    motion_field = cv2.line(motion_field, (a, b), (c, d), (60, 60, 60), 2)

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            cv2.imwrite(motion_file, motion_field)

        if frame_zero is None or motion_field is None:
            print(f'Warning: {img_path} seems to be a NoneObject!')

        # preprocess (convert to grayscale, to pytorch tensor and normalize)
        frame_zero = self.transform(frame_zero)
        motion_field = self.transform(motion_field)
        kind = re.search(r'(\d+)_(\w+)_(\w+)', img_path).group(3)
        motion_type = re.search(r'(\d+)_(\w+)_(\w+)', img_path).group(2)

        return frame_zero, motion_field, kind, motion_type, img_path

    def pick_a_pair(self):
        tok = random.random()
        cats = list(self.categories.keys())

        if tok < 0.30:  # pick an identical pair
            cat = random.choice(cats)
            chosen_os = random.sample(self.categories[cat], k=2)

            label = 1
        else:
            if tok < 0.55:  # pick a totally random pair
                chosen_cats = random.sample(cats, k=2)
                chosen_os = [random.choice(self.categories[cat]) for cat in chosen_cats]

            elif tok < 0.75:  # pick a pair with same org type and different actions
                if random.random() < 0.5:  # edible
                    chosen_cats = random.sample(cats[:3], k=2)
                    chosen_os = [random.choice(self.categories[cat]) for cat in chosen_cats]
                else:  # poisonous
                    chosen_cats = random.sample(cats[3:], k=2)
                    chosen_os = [random.choice(self.categories[cat]) for cat in chosen_cats]

            else:  # pick a pair with different org type and same actions
                if random.random() < 0.33:  # still action
                    chosen_cats = random.sample((1, 4), k=2)
                    chosen_os = [random.choice(self.categories[cat]) for cat in chosen_cats]

                elif random.random() < 0.66:  # spin action
                    chosen_cats = random.sample((2, 5), k=2)
                    chosen_os = [random.choice(self.categories[cat]) for cat in chosen_cats]

                else:  # movement action
                    chosen_cats = random.sample((3, 6), k=2)
                    chosen_os = [random.choice(self.categories[cat]) for cat in chosen_cats]

            label = 0

        return chosen_os[0], chosen_os[1], label

    def pick_samples_from_each_cat(self, n_samples):
        return {
            1: random.sample(self.categories[1], k=n_samples),
            2: random.sample(self.categories[2], k=n_samples),
            3: random.sample(self.categories[3], k=n_samples),
            4: random.sample(self.categories[4], k=n_samples),
            5: random.sample(self.categories[5], k=n_samples),
            6: random.sample(self.categories[6], k=n_samples),
        }

    def __getitem__(self, idx):
        if not self.return_pair:

            return self.preprocess(img_path=self.organisms[idx])
        else:
            org_1, org_2, label = self.pairs[idx]
            post_org_1 = self.preprocess(org_1)
            post_org_2 = self.preprocess(org_2)

            return post_org_1, post_org_2, label

    def __len__(self):
        if not self.return_pair:

            return len(self.organisms)
        else:

            return len(self.pairs)


if __name__ == '__main__':
    organism_ds = OrganismsDataset()
    # fig = plt.figure()
    #
    # for i in range(len(organism_ds)):
    #     sample = organism_ds[i][0]
    #     sample = sample.permute(2, 1, 0).numpy()
    #
    #     ax = plt.subplot(1, 4, i + 1)
    #     ax.set_title('Organism #{}'.format(i))
    #     ax.axis('off')
    #     plt.tight_layout()
    #     plt.imshow(sample.squeeze())
    #
    #     if i == 3:
    #         plt.show()
    #         break
    # from agents import *
    #
    # agent_test = SilentAgent()
    #
    # loader = DataLoader(organism_ds, batch_size=32, pin_memory=True)
    #
    # for batch_ndx, sample in enumerate(loader):
    #     print(sample)
