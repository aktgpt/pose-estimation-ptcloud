import csv
from math import pi
import argparse
import itertools
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import copy


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
    potential_match = PointsMatcher(known_points, observed_points)
    potential_matches = potential_match.get_potential_correspondences()
    pose_estimator = PoseEstimator(known_points, observed_points, potential_matches)
    pose = pose_estimator.get_transform()
    print(pose)

class PointsMatcher:
    def __init__(self, known_points, observed_points):
        self.known_points = known_points
        self.observed_points = observed_points
        # self.dmat = []
        # self.amat = []
        self.correspondence_matrix = self.generate_correspondence_matrix(known_points, observed_points)

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
        points = np.asarray(points)
        eps = 1e-3
        for subset in itertools.combinations(range(len(points)), 3):
            cross_prod = np.linalg.norm(np.cross(points[subset[0]]-points[subset[1]],
                                  points[subset[0]]-points[subset[2]]))
            if cross_prod < eps:
                pass
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
        # self.dmat.append(d)
        alpha_l = 1
        shape_param = np.exp(-d/alpha_l)
        return shape_param

    def surface_parameter(self, triangle_area_1, triangle_area_2):
        # self.amat.append((triangle_area_1-triangle_area_2)**2)
        alpha_s = 10
        surface_param = np.exp(-(triangle_area_1-triangle_area_2)**2 / alpha_s)
        return surface_param

    def get_potential_correspondences(self):
        corres_mat = copy.deepcopy(self.correspondence_matrix)
        potential_matches = []
        search_times = 2
        if corres_mat.shape[0] <= corres_mat.shape[1]:
            while search_times > 0:
                local_maximas = []
                for i in range(corres_mat.shape[0]):
                    potentialRightIdx = np.argmax(corres_mat[i, :])
                    potentialLeftIdx = np.argmax(corres_mat[:, potentialRightIdx])
                    if potentialLeftIdx == i:
                        local_maximas.append([potentialLeftIdx, potentialRightIdx])
                for i in range(len(local_maximas)):
                    corres_mat[local_maximas[i][0], local_maximas[i][1]] = 0
                search_times -= 1
                potential_matches.append(local_maximas)
        else:
            while search_times > 0:
                local_maximas = []
                for i in range(corres_mat.shape[1]):
                    potentialLeftIdx = np.argmax(corres_mat[:, i])
                    potentialRightIdx = np.argmax(corres_mat[potentialLeftIdx, :])
                    if potentialRightIdx == i:
                        local_maximas.append([potentialLeftIdx, potentialRightIdx])
                for i in range(len(local_maximas)):
                    corres_mat[local_maximas[i][0], local_maximas[i][1]] = 0
                search_times -= 1
                potential_matches.append(local_maximas)

        # generate subsets of matches
        matches = []
        for i in range(len(potential_matches)):
            matches_subset = []
            if len(potential_matches[i]) > 3:
                for subset in itertools.combinations(range(len(potential_matches[i])), len(potential_matches[i])-1):
                    for i in range(len(potential_matches[i])):
                        match_subset = [potential_matches[i][j] for j in subset]
                        matches_subset.append(match_subset)
            elif len(potential_matches[i]) == 3:
                matches_subset.append(potential_matches[i])
        matches.extend(matches_subset)
        return matches


class PoseEstimator:
    def __init__(self, known_points, observed_points, potential_matches):
        self.known_points = known_points
        self.observed_points = observed_points
        self.potential_matches = potential_matches

    def get_transform(self):
        errors = []
        Ts = []
        Rs = []
        ts = []
        for i in range(len(self.potential_matches)):
            known_combination = np.empty([len(self.potential_matches[i]), 3])
            observed_combination = np.empty([len(self.potential_matches[i]), 3])
            for j in range(len(self.potential_matches[i])):
                known_combination[j, :] = self.known_points[self.potential_matches[i][j][0]]
                observed_combination[j, :] = self.observed_points[self.potential_matches[i][j][1]]
            T, R, t = self.best_fit_transform(known_combination, observed_combination)
            error = self.get_transformation_loss(known_combination, observed_combination, T)
            errors.append(error)
            Ts.append(T)
            Rs.append(R)
            ts.append(t)
        min_error_idx = np.argmin(errors)
        T_best = Ts[min_error_idx]
        R_best = Rs[min_error_idx]
        t_best = ts[min_error_idx]

        return T_best, R_best, t_best

    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Nxm numpy array of corresponding points
          B: Nxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''
        if A.shape != B.shape:
            print('foo')

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def get_transformation_loss(self, A, B, transform):
        m = A.shape[1]
        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, A.shape[0]))
        dst = np.ones((m + 1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        src = np.dot(transform, src)

        # error
        distances = np.linalg.norm((src - dst), axis=0)
        mean_error = np.mean(distances)

        return mean_error

if __name__ == '__main__':
    args = argparser.parse_args(['-kp', 'tools/pose_estimation/known_markers.csv', '-op', 'tools/pose_estimation/observed_markers.csv'])
    _main_(args)