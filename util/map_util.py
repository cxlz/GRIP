import argparse
import glob
import logging
import os
import sys
root_path = os.path.abspath(".")
# print(root_path)
source_list = [os.path.join(root_path, 'util'), os.path.join(root_path, 'util/proto')]
for source in source_list:
    try:
        sys.path.append(source)
    except:
        pass

from typing import List
from util.proto import map_pb2, map_lane_pb2
import google.protobuf.text_format as text_format

class LaneSegment():
    def __init__(self):
        self.id = ""
        self.turn = -1
        self.segment = []

    def set_id(self, id):
        self.id = id

    def set_turn(self, turn_type):
        self.turn = turn_type

class HDMap:
    def __init__(self, map_file=""):
        self.map_file = map_file
        self.hdmap = map_pb2.Map()
        self.lane_dict = {}
        self.lane_boundary_dict = {}
        self.segment_point_num = 20
        if map_file:
            self.setup(map_file)


    def setup(self, map_file):
        self.get_pb_from_file(map_file, self.hdmap)

        
    def get_pb_from_text_file(self, filename, pb_value):
        """Get a proto from given text file."""
        with open(filename, 'r') as file_in:
            return text_format.Merge(file_in.read(), pb_value)


    def get_pb_from_bin_file(self, filename, pb_value):
        """Get a proto from given binary file."""
        with open(filename, 'rb') as file_in:
            pb_value.ParseFromString(file_in.read())
        return pb_value


    def get_pb_from_file(self, filename, pb_value):
        """Get a proto from given file by trying binary mode and text mode."""
        try:
            return self.get_pb_from_bin_file(filename, pb_value)
        except:
            try:
                return self.get_pb_from_text_file(filename, pb_value)
            except:
                print('Error: Cannot parse %s as binary or text proto' % filename)
        return None


    def get_lanes(self, center_point, search_radius, point_num=10) -> List[LaneSegment]:
        lane_list = []
        s_distance = search_radius ** 2
        if self.segment_point_num != point_num:
            self.lane_dict.clear()
            self.segment_point_num = point_num
        for lane in self.hdmap.lane:
            if lane.turn == 4:
                continue
            lane_id = lane.id.id
            lane_segment = LaneSegment()
            lane_segment.set_id(lane_id)
            lane_segment.set_turn(lane.turn)
            if lane_id not in self.lane_dict.keys():
                self.build_lane_segment(lane)
            for seg in self.lane_dict[lane_id].segment:
                for point in seg:
                    if self.square_distance(center_point, point) < s_distance:
                        lane_segment.segment.append(seg)
                        break
            if len(lane_segment.segment) > 0:
                lane_list.append(lane_segment)
        return lane_list

            
    def build_lane_segment(self, lane):
        lane_segment = LaneSegment()
        lane_id = lane.id.id
        lane_segment.set_id(lane_id)
        lane_segment.set_turn(lane.turn)
        if lane.HasField("central_curve"):
            curr_segment = []
            for seg in lane.central_curve.segment:
                if seg.HasField("line_segment"):
                    points = seg.line_segment.point
                    for i in range(0, len(points), 3):
                        p = points[i]
                        curr_segment.append([p.x, p.y])
                        if len(curr_segment) == self.segment_point_num or i == len(points) - 1:
                            lane_segment.segment.append(curr_segment)
                            curr_segment = [[p.x, p.y]]
        self.lane_dict[lane_id] = lane_segment


    def square_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


    def get_lane_boundary(self, lane:map_lane_pb2.Lane()):
        lane_id = lane.id.id
        if lane_id not in self.lane_boundary_dict:
            max_x = float("-inf")
            max_y = float("-inf")
            min_x = float("inf")
            min_y = float("inf")
            if lane.HasField("central_curve"):
                for seg in lane.central_curve.segment:
                        if seg.HasField("line_segment"):
                            curve_point_list = []
                            for p in seg.line_segment.point:
                                max_x = max(max_x, p.x)
                                max_y = max(max_y, p.y)
                                min_x = min(min_x, p.x)
                                min_y = min(min_y, p.y)
                self.lane_boundary_dict[lane_id] = [max_x, min_x, max_y, min_y]
        return self.lane_boundary_dict[lane_id]




if __name__ == "__main__":
    root_path = os.path.abspath("./")
    map_file = "data/map/huaxin/base_map.bin"
    print(os.path.isfile(map_file))
    hdmap = HDMap(os.path.join(root_path, map_file))
    hdmap.get_lanes([0, 0], 50)
