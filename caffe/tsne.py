#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import random

def get_cmap(N):
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def c_index(N):
    return 0.5*(N)*(N+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('t-SNE visualization generator')
    parser.add_argument('-i', '--input', help='path to input feature csv file with last column as labels', required=True, type=str)
    parser.add_argument('-o', '--output', help='Output pdf or png file', required=True, type=str)
    parser.add_argument('-d', '--dpi', help='DPI for output, default 300', required=False, type=int, default=300)
    parser.add_argument('-r', '--ratio', help='Ratio of total points uniformly sampled (default 1)', required=False, type=float, default=1)
    parser.add_argument('-c', '--classes', help='Randomly select classes (default -1)', required=False, default=-1, type=float)
    args = parser.parse_args()

    XY = np.genfromtxt(args.input, delimiter=',')
    rand_matrix = np.random.randint(XY.shape[0], size = int(args.ratio*XY.shape[0]))
    X = XY[rand_matrix,:-1]
    y = XY[rand_matrix,-1].astype(int)

    if args.classes < 0:
        n = len(set(list(y)))
    else:
        n = int(args.classes)
    crange = range(len(set(list(y))))
    random.shuffle(crange)
    crange = crange[:n]

    cmap = get_cmap(c_index(n))
    tsne = TSNE(n_components=2)
    X_red = tsne.fit_transform(X)

    xmin = np.min(X_red[:,0])
    xmax = np.max(X_red[:,0])
    ymin = np.min(X_red[:,1])
    ymax = np.max(X_red[:,1])

    print 'TSNE completed, writing points now'

    for xx, yx in zip(X_red, y):
        if int(yx) in crange:
            plt.scatter(xx[0], xx[1], color=cmap(c_index(crange.index(yx))))

    print 'Plotting done, saving image'

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.savefig(args.output, dpi=args.dpi)




