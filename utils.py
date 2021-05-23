# plotly.offline.init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import torch.nn as nn


class IDGen:
    def __init__(self):
        self.ids = [0]

    @property
    def new(self):
        self.ids.append(max(self.ids) + 1)
        return self.ids[-1]

    @property
    def curr(self):
        return self.ids[-1]


def plot_tSNE(pt, Y, title):
    """
    To plot a 3D plot using plotly. (For MNIST data only)

    Parameters:
        pt: points in 3 dimensions with corresponding labels/names in "Y"
        Y: labels/names for each point in "pt"
        title: For title of plot
    Output:
        plots a 3D interactive graph of points in "pt" having labels in "Y"
    """
    data = []
    for i in range(10):
        index = (Y == i)
        trace = go.Scatter3d(x=pt[index, 0], y=pt[index, 1], z=pt[index, 2], mode='markers',
                             marker=dict(size=6, line=dict(color=plt.cm.Set1(i / 10.)), opacity=0.97),
                             text=[f'{i}'], name=f'{i}', hoverinfo='name')
        data.append(trace)
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0), title=title)
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename=title)


def show(img, title=None):
    """
    Function to plot a single image.
    -------------------------------------------------
    Parameters:
        img: A numpy array of image
        title: Title for plot
    Output:
        Plots image using matplotlib.
    """
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


def plots(ims, figsize=(12, 6), rows=2, titles=None):
    """
    Plot multiple images in diff. subplots, configured by parameter "rows".
    ----------------------------------------------------------------------------
    Parameters:
        ims: An numpy array of arrays of diff images
        figsize: parameter to be passed to matplotlib "plt" function
        rows: number of rows in plot for subplots
        titles: Array of titles for all images
    Output:
        Plot a matplotlib plot with (r*c) subplots
    """
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def init_weights(m):
    for name, param in m.named_parameters():
        if 'Conv' not in name:
            nn.init.uniform_(param.data, -0.08, 0.08)


def get_message(s, vocab_size):
    return ''.join([chr(97 + int(v.cpu().data)) for v in s if v < vocab_size])


def convert_to_chars(raw_sen):
    return ''.join([chr(int(i) + 96) for i in raw_sen.argmax(1).tolist()[0]])


def print_round_stats(acc, sl, loss):
    print("*******")
    print("Round average accuracy: %.2f" % (acc * 100))
    print("Round average sentence length: %.1f" % sl)
    print("Round average loss: %.1f" % loss)
    print("*******")
