import csv
import cairo
from math import pi
import argparse
import itertools
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser(
    description='Annotate images in a folder')

argparser.add_argument(
    '-kp',
    '--known_points',
    help='path to known points')
argparser.add_argument(
    '-op',
    '--observed_points',
    help='path to observed points')


def read_points_location(points_location):
    with open(points_location, 'r') as f:
        reader = csv.reader(f)
        points2D = list(reader)
    points2Dint = []
    for i in range(len(points2D)):
        points2Dint.append([float(x) for x in points2D[i]])
    return points2Dint

def _main_(args):
    known_points_path = args.known_points
    observed_points_path = args.observed_points
    known_points = read_points_location(known_points_path)
    observed_points = read_points_location(observed_points_path)
    pose = PointsMatcher(known_points, observed_points)

class PointsMatcher:
    def __init__(self, known_points, observed_points):
        self.known_points = known_points
        self.observed_points = observed_points
        self.dmat = []
        self.amat = []
        corres_mat = self.generate_correspondence_matrix(known_points, observed_points)
        poten_matches = self.get_potential_correspondences(corres_mat)

    def generate_correspondence_matrix(self, known_points, observed_points):
        known_triangle_idx, known_triangle_sides, known_areas = self.triangle_list_idx(known_points)
        observed_triangle_idx, observed_triangle_sides, observed_areas = self.triangle_list_idx(observed_points)
        correspondance_matrix = np.zeros([len(known_points), len(observed_points)])
        for i in range(len(known_triangle_idx)):
            for j in range(len(observed_triangle_idx)):
                similarity_score = self.shape_parameter(known_triangle_sides[i], observed_triangle_sides[j]) * \
                                    self.surface_parameter(known_areas[i], observed_areas[j])
                for k in range(len(known_triangle_idx[i])):
                    correspondance_matrix[known_triangle_idx[i][k], observed_triangle_idx[j][k]] += similarity_score
        correspondance_matrix = correspondance_matrix / np.linalg.norm(correspondance_matrix)
        return correspondance_matrix

    def triangle_list_idx(self, points):
        distance_mat = distance.cdist(points, points, 'sqeuclidean')
        triangle_lists_idx = []
        triangle_sides = []
        triangle_areas = []
        for subset in itertools.combinations(range(len(points)), 3):
            # TODO: add co-linearity condition
            triangle_side_lengths = [distance_mat[subset[0], subset[1]], distance_mat[subset[1], subset[2]],
                                     distance_mat[subset[0], subset[2]]]
            # TODO: add for isoceles case
            sorted_sides_idx = np.argsort(triangle_side_lengths)
            subset_corrected = [subset[sorted_sides_idx[i]] for i in range(len(sorted_sides_idx))]
            triangle_sides.append(sorted(triangle_side_lengths))
            triangle_lists_idx.append(subset_corrected)
            triangle_areas.append(self.area_from_points([points[subset[0]], points[subset[1]],
                                    points[subset[2]]]))
        return triangle_lists_idx, triangle_sides, triangle_areas

    def area_from_points(self, points):
        points = np.asarray(points)
        ab = points[1]-points[0]
        ac = points[2]-points[0]
        area = 0.5 * np.linalg.norm(np.cross(ab, ac))
        return area

    def shape_parameter(self, triangle_sides_1, triangle_sides_2):
        d = (triangle_sides_1[0]-triangle_sides_2[0])**2 + (triangle_sides_1[0]-triangle_sides_2[0])**2 \
            + (triangle_sides_1[0]-triangle_sides_2[0])**2
        self.dmat.append(d)
        alpha_l = 1
        shape_param = np.exp(-d/alpha_l)
        return shape_param

    def surface_parameter(self, triangle_area_1, triangle_area_2):
        self.amat.append((triangle_area_1-triangle_area_2)**2)
        alpha_s = 10
        surface_param = np.exp(-(triangle_area_1-triangle_area_2)**2 / alpha_s)
        return surface_param

    def get_potential_correspondences(self, correspondence_matrix):
        matches_list = []
        if correspondence_matrix.shape[0] <= correspondence_matrix.shape[1]:
            matches = np.empty([correspondence_matrix.shape[0], 2], dtype=int)
            for i in range(correspondence_matrix.shape[0]):
                matches[i, :] = (i, np.argmax(correspondence_matrix[i, :]))
        else:
            matches = np.empty([correspondence_matrix.shape[1], 2], dtype=int)
            for i in range(correspondence_matrix.shape[1]):
                matches[i, :] = (np.argmax(correspondence_matrix[:, i]), i)
        return matches


if __name__ == '__main__':
    args = argparser.parse_args(['-kp', 'known_markers.csv', '-op', 'observed_markers.csv'])
    _main_(args)