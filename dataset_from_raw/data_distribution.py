import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load import load_all

def get_coordinate_frame(data,metadata):
    # coordinate frame for each captured image frame
    origin = data_by_marker_name("ZeroPoint", data, metadata)
    x_axis = data_by_marker_name("XaxisPlus", data, metadata)
    y_axis = data_by_marker_name("YaxisPlus", data, metadata)
    e_x = (x_axis - origin) / np.linalg.norm(x_axis - origin, axis=1)[...,None]
    e_y = (y_axis - origin) / np.linalg.norm(y_axis - origin, axis=1)[...,None]

    normal = np.cross(e_x,e_y)
    normal_length = np.linalg.norm(normal, axis=1)
    normal = normal / normal_length[...,None]

    return origin, e_x, e_y, normal

def row(marker_name, metadata):
    marker_idx = [i for i, x in enumerate(metadata["MARKER_NAMES"]) if x == marker_name]
    base_position = np.array([0,1,2], dtype=int)

    # +2 offset because Frame and Time labels are in COLUMN_NAMES but not in MARKER_NAMES
    positions = base_position[None,...] + 3*np.array(marker_idx)[...,None] + 2

    return positions

def average_multiple_columns(column_idx, data):
    """If multiple markers for a position exist, average them
        @return mean_data   the data averaged over all markers excluding 0 entries
        @return unusable    index along the first dimension, True if only 0 entries 
                            were found for the entire data
    """
    eps = 1e-8

    column_data = np.stack([data[:,point] for point in column_idx],axis=2)
    column_data[column_data < eps] = np.nan
    mean_data = np.nanmean(column_data, axis=2)
    return mean_data

def data_by_marker_name(marker_name, data, metadata):
    rows = row(marker_name, metadata)
    if rows.shape[0] == 0:
        # no data available because the marker could not be found
        # return empty data instead
        ret_val = np.empty((data.shape[0], 3))
        ret_val[:] = np.nan
        return ret_val
    mean_data = average_multiple_columns(rows, data)
    return mean_data

def participant_height(data,metadata):
    # coordinate frame for each captured image frame
    _, _, _, normal = get_coordinate_frame(data,metadata)

    head_position = data_by_marker_name("UsersHead", data, metadata)
    if head_position.shape[0] > 0:
        height = np.sum(head_position * normal, axis=1)
    else:
        height = 0.0

    return height

def get_comfort(data, metadata):
    # coordinate frame for each captured image frame
    _, _, _, normal = get_coordinate_frame(data, metadata)

    hand_position = data_by_marker_name("UsersHand", data, metadata)
    
    return np.sum(hand_position * normal, axis=1)


def marker_projection(marker_name, data, metadata):
    """returns the selected marker projected into the used 2D referece frame"""

    # coordinate frame for each captured image frame
    origin, e_x, e_y, _ = get_coordinate_frame(data, metadata)

    marker_position_3d = data_by_marker_name(marker_name, data, metadata)
    marker_vector = marker_position_3d - origin

    # project the data
    marker_x = np.sum(np.multiply(marker_vector, e_x), axis=1)
    marker_y = np.sum(np.multiply(marker_vector, e_y), axis=1)

    return marker_x, marker_y

def get_distance(data, metadata):
    robot_markers = ["DoubleBottom","DoubleFaceBottomRight","DoubleFaceTopRight","DoubleFaceTopLeft","DoubleFaceBottomLeft"]
    robot_data_list = list()

    for marker in robot_markers:
        marker_data = data_by_marker_name(marker, data, metadata)
        robot_data_list.append(marker_data)
    robot_data = np.stack(robot_data_list, axis=2)
    robot_data = np.nanmean(robot_data, axis=2)

    human_data = data_by_marker_name("UsersHead", data, metadata)

    distance = np.linalg.norm(robot_data - human_data, axis=1)
    return distance

if __name__ == "__main__":
    by_id, by_position = load_all("../tsv_files/*.tsv")

    #plot comfort
    #"""
    def plot_comfort(trial):
        metadata = trial["metadata"]
        data = trial["data"]
        eps = 1e-8

        comfort =  get_comfort(data,metadata)
        plt.plot(comfort)

    user_data = by_id[4]
    for position in ["A","B","C","D", "E"]:
        plot_comfort(user_data[position])
    plt.show()
    #"""
    #-----

    #polar plot position
    """
    def polar_plot(trial, axis):
        metadata = trial["metadata"]
        data = trial["data"]
        eps = 1e-8

        head_x, head_y = marker_projection("UsersHead", data, metadata)
        head = np.stack((head_x,head_y),axis=1)
        zero_head = np.any(head < eps, axis=1)

        robot_markers = ["DoubleBottom","DoubleFaceBottomRight","DoubleFaceTopRight","DoubleFaceTopLeft","DoubleFaceBottomLeft"]
        for marker in robot_markers:
            marker_x, marker_y = marker_projection(marker, data, metadata)
            marker_data = np.stack((marker_x,marker_y), axis=1)
            zero_marker = np.any(marker_data < eps, axis=1)

            usable = np.logical_not(np.logical_or(zero_head,zero_marker))

            distance_vector = marker_data[usable,:] - head[usable,:]
            theta = np.arctan2(distance_vector[:,1],distance_vector[:,0])
            distance = np.linalg.norm(distance_vector, axis=1)

            ax.plot(theta, distance)

    user_data = by_id[3]
    ax = plt.subplot("111", projection="polar")
    ax.set_rlabel_position(315)
    ax.grid(True)
    for position in ["A","B","C","D", "E"]:
        polar_plot(user_data[position], ax)
    plt.show()
    """
    #-----

    #plot height
    """
    def plot_height(trial):
        metadata = trial["metadata"]
        data = trial["data"]

        height = participant_height(data, metadata)
        plt.plot(height)

    user_data = by_id[19]
    for position in ["A","B","C","D", "E"]:
        plot_height(user_data[position])

    plt.ylabel("Height (in mm)")
    plt.xlabel("Frame")
    plt.show()
    """
    #-----

    #plot distance
    """
    def plot_distance(trial):
        metadata = trial["metadata"]
        data = trial["data"]
        eps = 1e-8

        frame = data[:,0]

        head_x, head_y = marker_projection("UsersHead", data, metadata)
        head = np.stack((head_x,head_y),axis=1)
        zeros = np.any(head < eps, axis=1)
        if np.any(zeros):
            print("Head Undetectable for some frames")

        robot_markers = ["DoubleBottom","DoubleFaceBottomRight","DoubleFaceTopRight","DoubleFaceTopLeft","DoubleFaceBottomLeft"]
        for marker in robot_markers:
            marker_x, marker_y = marker_projection(marker, data, metadata)
            marker_data = np.stack((marker_x,marker_y), axis=1)
            distance = np.linalg.norm(marker_data - head, axis=1)
            zero_elements = np.logical_or(marker_x < eps, marker_y < eps)
            plt.plot(frame-frame[0], distance)

    user_data = by_id[8]
    for position in ["A","B","C","D", "E"]:
        plot_distance(user_data[position])
    plt.show()

    """
    #-----

    # proxemics distribution
    """
    def plot_approach(trial):
        metadata = trial["metadata"]
        data = trial["data"]
        eps = 1e-8

        head_x, head_y = marker_projection("UsersHead", data, metadata)
        plt.plot(head_x, head_y)
        zero_elements = np.logical_or(head_x < eps, head_y < eps)
        plt.plot(head_x[np.logical_not(zero_elements)], head_y[np.logical_not(zero_elements)])

        robot_markers = ["DoubleBottom","DoubleFaceBottomRight","DoubleFaceTopRight","DoubleFaceTopLeft","DoubleFaceBottomLeft"]
        for marker in robot_markers:
            marker_x, marker_y = marker_projection(marker, data, metadata)
            zero_elements = np.logical_or(marker_x < eps, marker_y < eps)
            plt.plot(marker_x[np.logical_not(zero_elements)], marker_y[np.logical_not(zero_elements)])


    user_data = by_id[8]   
    for position in ["A","B","C","D", "E"]:
        plot_approach(user_data[position])
    plt.show()
    """
    #-----

    # number of zero points after averaging over data with the same name
    """
    robot_markers = ["DoubleBottom","DoubleFaceBottomRight","DoubleFaceTopRight","DoubleFaceTopLeft","DoubleFaceBottomLeft"]
    head_markers = ["UsersHead"]
    hand_markers = ["UsersHand"]

    labels_to_check = robot_markers + head_markers + hand_markers
    positions = ["A","B","C","D","E"]
    discardable_by_position = np.zeros((5,7), dtype=int)
    total_by_position = np.zeros((5,7), dtype=int)
    for position, person_data in by_position.items():
        for user, trial in enumerate(person_data):
            data = trial["data"]
            metadata = trial["metadata"]
            discardable_data = list()
            total_data = list()
            for marker in labels_to_check:
                _, unusable = data_by_marker_name(marker, data, metadata)
                discardable_data.append(np.sum(unusable))
                total_data.append(unusable.shape[0])
            if np.sum(discardable_data) > 0:
                print("User %d has data missing for direction %s" % (user, position))
            discardable_by_position[positions.index(position),:] += discardable_data
            total_by_position[positions.index(position),:] += total_data
    usable_percent = 1 - np.divide(discardable_by_position,total_by_position)
    plt.imshow(usable_percent)
    for (j,i),label in np.ndenumerate(usable_percent):
        plt.gca().text(i,j,np.around(usable_percent[j,i],3),ha='center',va='center')
    plt.gca().set_yticks(range(0,5))
    plt.gca().set_yticklabels(positions)
    plt.gca().set_xticks(range(0,7))
    plt.gca().set_xticklabels(["Base","BRight","URight","ULeft","BLeft","Head","Hand"])

    plt.show()
    """
    #-----

    #plot distribution over origin
    """
    origin_data = np.empty((0,3))
    for position, person_data in by_position.items():
        for trial in person_data:
            data, unusable = data_by_marker_name("ZeroPoint", trial["data"],trial["metadata"])
            origin_data = np.concatenate((origin_data,data[unusable == False]),axis=0)
    origin_data[np.isnan(origin_data)] = 0.0
    print(np.any(np.isnan(origin_data[:,0])))
    plt.hist(origin_data[:,0].tolist())
    plt.show()
    """
    #-----

    # total count of zeros in the entire dataset
    """
    total_count = dict()
    total_zero = dict()
    for position in by_position:
        total_count[position] = 0.0
        total_zero[position] = 0.0

    for position, data in by_position.items():
        for user_data in data:
            data = user_data["data"]
            metadata = user_data["metadata"]
            total_zero[position] += np.sum(data < 0.00001)
            total_count[position] += np.prod(data.shape)
        
    print([total_zero[position]/total_count[position] for position in total_count])
    """
    #-----