import enum
import os
import numpy as np
import matplotlib.pyplot as plt


class Graspability():
    def __init__(self):
        self.radius = 20 #TODO: tune
        self.trace_thresh = 100 #TODO: tune

    def find_trace_dist(self, point1, point2, pixel_point_to_idx ):
        return abs(pixel_point_to_idx[point1] - pixel_point_to_idx[point2])

    def find_pixel_point_graspability(self, point, trace):
            total_points = 4*(self.radius**2)
            points_outside_trace_threshold = 0
            start_point_x = point[0] - self.radius
            start_point_y = point[1] - self.radius

            for i in range(start_point_x, start_point_x + 2*self.radius):
                for j in range(start_point_y, start_point_y  + 2*self.radius):
                    neigh =(i,j)
                    if neigh in trace:
                        #if the neighboring pixel is in the rope trace and is within 
                        if(self.find_trace_dist(point, neigh, trace) > self.trace_thresh):
                            points_outside_trace_threshold += 1
            return points_outside_trace_threshold / total_points