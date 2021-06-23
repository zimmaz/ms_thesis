import re
from glob import iglob
from random import uniform, randint

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib as mpl
import numpy as np
from matplotlib import transforms
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

global base

mpl.rc('hatch', color='k', linewidth=1.5)


def create_voronoi(vor: Voronoi, radius: float = None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates of input vertices, with 'points at infinity'
        appended to the end.

    --------
    function taken from https://stackoverflow.com/a/20678647/12473863 with minor modifications.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


# noinspection PyGlobalUndefined
class OrganismGenerator:
    def __init__(self, path='./organisms/'):
        self.path = path
        self.min_cell_no = 20
        self.max_cell_no = 120
        self.noise = True
        # check if any organism already exists in the `path` directory
        existing_organisms = [
            int(re.search(r'(\d+)_(\w+)_(\w+)', o).group(1)) for o in iglob(f'{self.path}/*.gif')
        ]
        # get the latest/highest ID amongst the organisms
        self.last = max(existing_organisms) if existing_organisms else 0

    def generate(self, cellular_density):
        no_of_cells = int(self.min_cell_no + cellular_density * (self.max_cell_no - self.min_cell_no))
        # get initial random 2d points from a normal dist
        points = np.random.normal(
            loc=(uniform(0.0, 60.0), uniform(0.0, 60.0)),
            scale=uniform(0.0, 30.0),
            size=(no_of_cells, 2)
        )
        # create Voronoi
        vor = Voronoi(points)
        regions, vertices = create_voronoi(vor)

        min_x = vor.min_bound[0] - 0.1
        max_x = vor.max_bound[0] + 0.1
        min_y = vor.min_bound[1] - 0.1
        max_y = vor.max_bound[1] + 0.1

        box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        cells = []
        nucleoids = []

        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = [p for p in poly.boundary.coords]
            # check if the coords are inside the box
            x_ins = [min_x < x < max_x for x, y in polygon]
            y_ins = [min_y < y < max_y for x, y in polygon]

            if all(x_ins) and all(y_ins):
                # add the centroid of each polygon (nucleoids)
                centroid = np.mean(polygon, axis=0)
                cells.append(np.array(polygon))
                nucleoids.append(centroid)

        return cells, np.array(nucleoids)

    def draw(self, idx_: int, org_type: str, movement: str, taste: str, dpi=10, n_frames=10):
        if org_type == 'edible':
            density = np.random.choice(np.arange(0.1, 0.3, 0.1), 1)[0]
        else:
            density = np.random.choice(np.arange(0.7, 1, 0.1), 1)[0]

        # select a random color from Blues/Reds/Greens color maps
        # if color == 'blue':
        #     cm = plt.cm.get_cmap('Blues', 25)  # 25 distinct colors from blue spectrum and so on
        # elif color == 'green':
        #     cm = plt.cm.get_cmap('Greens', 25)
        # else:
        #     cm = plt.cm.get_cmap('Reds', 25)

        if taste == 'sweet':
            hatch = 'o'
        elif taste == 'bitter':
            hatch = '+'
        else:
            hatch = '/'

        # choose a random color from the spectrum
        # global col_idx
        # global chosen_color
        # col_idx = randint(15, 25)  # only from the second half of the color spectrum (for better visibility)
        # chosen_color = cm(col_idx)

        # pick a random color for background
        bg_color = plt.cm.get_cmap('Pastel1', 8)(randint(0, 9))
        cells_, nucleoids_ = self.generate(cellular_density=density)

        # choose spin direction and movement randomly
        sign = np.random.choice([-1, 1], 1)[0]
        tx_ = np.random.choice(np.arange(-10, 10), 1)[0]
        ty_ = np.random.choice(np.arange(-10, 10), 1)[0]

        # get a random hatch to add some noise to organism
        # hatches = ['//', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']
        # if self.noise:
        #     hatch = np.random.choice(hatches, 1)[0]
        # else:
        #     hatch = None

        # choose a color map and pick colors for cells
        cm = plt.cm.get_cmap('tab20c', 20)
        cell_cols = [cm(randint(0, 20)) for _ in cells_]

        alpha = uniform(0.5, 1.0)  # transparency
        linewidth = uniform(3, 10)

        plt.figure(figsize=(6.4, 6.4), dpi=dpi)

        # this function loads the initial frame (frame zero) in the animation
        def init():
            dots_to_draw_x = [d[0] for d in nucleoids_]
            dots_to_draw_y = [d[1] for d in nucleoids_]
            plt.plot(dots_to_draw_x, dots_to_draw_y, 'o', markersize=20)

            for c, c_col in zip(cells_, cell_cols):
                c_pol = ptc.Polygon(c, closed=True, hatch=hatch, alpha=alpha, linewidth=linewidth, edgecolor='k', facecolor=c_col)
                # plt.fill(*zip(*c), alpha=alpha, hatch=hatch)
                # plt.plot(c_pol, linewidth=8, color='k')
                plt.gca().add_patch(c_pol)
            global base
            base = plt.gca().transData

        # make a function to switch to next frame in FuncAnimation
        def plot_next(n):
            plt.cla()
            plt.axis('off')
            plt.tight_layout()

            if movement == 'spin':
                offset = transforms.Affine2D().rotate_deg(n * 36 * sign) + base
            elif movement == 'move':
                offset = base + transforms.Affine2D().translate(tx_ * n, ty_ * n)
            else:
                offset = base

            dots_to_draw_x = [d[0] for d in nucleoids_]
            dots_to_draw_y = [d[1] for d in nucleoids_]
            plt.plot(dots_to_draw_x, dots_to_draw_y, 'o', markersize=uniform(8, 13), transform=offset, color='k')

            for c, c_col in zip(cells_, cell_cols):
                if movement == 'camou':
                    c_pol = ptc.Polygon(c, closed=True, hatch=hatch, alpha=alpha, linewidth=linewidth, transform=offset, edgecolor='k', facecolor=cm(randint(0, 20)))
                    # plt.fill(*zip(*c), transform=offset, alpha=alpha, hatch=hatch, color=camou_cm(randint(1, 50)))
                else:
                    c_pol = ptc.Polygon(c, closed=True, hatch=hatch, alpha=alpha, linewidth=linewidth, transform=offset, edgecolor='k', facecolor=c_col)
                    # plt.fill(*zip(*c), transform=offset, alpha=alpha, hatch=hatch)

                # plt.plot(c_pol, linewidth=linewidth, color='k', transform=offset)
                plt.gca().add_patch(c_pol)
        ani = animation.FuncAnimation(plt.gcf(), plot_next, n_frames, init_func=init, interval=600)
        ani.save(
            f'{self.path}{str(idx_).zfill(6)}_{movement}_{org_type}_{taste}.gif',
            dpi=dpi,
            writer='pillow',
            savefig_kwargs={'facecolor': bg_color}
        )
        plt.axis('equal')
        print(f'Item {idx_} generated as a {taste} {org_type} organism - {movement}!')
        plt.close()

    def draw_multiple(self, n_of_organisms):
        for id_ in range(self.last + 1, self.last + n_of_organisms + 1):
            try:
                kind = np.random.choice(['edible', 'poisonous'], 1)[0]
                action = np.random.choice(['camou', 'move', 'spin', 'still'], 1)[0]
                taste = np.random.choice(['sweet', 'bitter', 'sour'], 1)[0]
                self.draw(id_, kind, movement=action, taste=taste)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    # data for tuning
    # mother = OrganismGenerator(path='./organisms/')
    # mother.draw_multiple(2400)
    # data for testing
    # mother = OrganismGenerator(path='./organisms/test/')
    # mother.draw_multiple(600)
    # data only used for tuning the visual module on spatial features
    mother = OrganismGenerator(path='./organisms/ft_data/test/')
    mother.draw_multiple(300)
    pass



    # mother.draw(1, 'poisonous', 'camou', 'sour')
    # mother.draw(2, 'poisonous', 'move', 'b')
    # mother.generate_specie_b(0.8)
    # fig, ax = plt.subplots()
    # # ax.set_aspect("equal")
    #
    # rad = 0.2
    # edgy = 0.05
    #
    # for c in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):
    #     a = get_random_points(n=10, scale=1) + c
    #     x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    #     plt.plot(x, y)
    #
    # plt.show()
