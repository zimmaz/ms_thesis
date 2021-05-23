import re
from glob import iglob
from random import uniform, randrange

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

global base


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
        points = np.random.normal(loc=0, scale=0.4, size=(no_of_cells, 2))
        # create Voronoi
        vor = Voronoi(points)
        regions, vertices = create_voronoi(vor)

        min_x = vor.min_bound[0] - 0.1
        max_x = vor.max_bound[0] + 0.1
        min_y = vor.min_bound[1] - 0.1
        max_y = vor.max_bound[1] + 0.1

        box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        cells_ = []
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
                cells_.append(np.array(polygon))
                nucleoids.append(centroid)

        return cells_, np.array(nucleoids)

    def draw(self, idx_: int, org_type: str, movement: str = 'move', dpi=10):
        if org_type == 'edible':
            density = np.random.choice(np.arange(0.1, 0.3, 0.1), 1)[0]
        else:
            density = np.random.choice(np.arange(0.7, 1, 0.1), 1)[0]

        cells_, nucleoids_ = self.generate(cellular_density=density)
        plt.figure(figsize=(6.4, 6.4), dpi=dpi)

        # this function loads the initial frame (frame zero) in the animation
        def init():
            dots_to_draw_x = [d[0] for d in nucleoids_]
            dots_to_draw_y = [d[1] for d in nucleoids_]
            plt.plot(dots_to_draw_x, dots_to_draw_y, 'kH', markersize=20)

            for c in cells_:
                plt.plot(*zip(*c), linewidth=8, color='k')
                plt.fill(*zip(*c))

            global base
            base = plt.gca().transData

        # choose spin direction and movement randomly
        sign = np.random.choice([-1, 1], 1)[0]
        tx_ = np.random.choice(np.arange(-10, 10), 1)[0]
        ty_ = np.random.choice(np.arange(-10, 10), 1)[0]

        # get a random hatch to add some noise to organism
        hatches = ['//', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']
        if self.noise:
            hatch = np.random.choice(hatches, 1)[0]
        else:
            hatch = None

        alpha = uniform(0.5, 1.0)
        linewidth = uniform(3, 12)

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
            plt.plot(dots_to_draw_x, dots_to_draw_y, 'kH', markersize=uniform(8, 13), transform=offset)

            for c in cells_:
                plt.plot(*zip(*c), linewidth=linewidth, color='k', transform=offset)
                plt.fill(*zip(*c), transform=offset, alpha=alpha, hatch=hatch)

        ani = animation.FuncAnimation(plt.gcf(), plot_next, 10, init_func=init)
        print(f'Organism {idx_} generated')
        ani.save(f'{self.path}{str(idx_).zfill(6)}_{movement}_{org_type}.gif', dpi=dpi, writer='pillow')
        plt.close()

    def draw_multiple(self, n_of_organisms):
        for id_ in range(self.last + 1, self.last + n_of_organisms + 1):
            try:
                kind = np.random.choice(['edible', 'poisonous'], 1)[0]
                action = np.random.choice(['still', 'spin', 'move'], 1)[0]
                mother.draw(id_, kind, movement=action)
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    mother = OrganismGenerator(path='./organisms/test/')
    mother.draw_multiple(500)
