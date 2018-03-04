import numpy as np
import matplotlib.pyplot as plt

from load import load_all
from data_distribution import get_comfort, participant_height, marker_projection, get_distance

#load raw data
def generate_dataset():
    by_id, by_position = load_all("../tsv_files/*.tsv")

    positions = ["A","B","C","D","E"]
    dataset = np.empty((0,10)) # magic number, cuz I'm a wizzard ya know? =)
    for user, data in enumerate(by_id):
        for position, trial in data.items():
            metadata = trial["metadata"]
            raw_data = trial["data"]
            eps = 1e-8

            #frame counter
            frames = raw_data[:,0] - raw_data[0,0]

            #robot position
            robot_markers = ["DoubleBottom","DoubleFaceBottomRight","DoubleFaceTopRight","DoubleFaceTopLeft","DoubleFaceBottomLeft"]
            marker_positions = list()
            for marker in robot_markers:
                try:
                    marker_position = np.stack(marker_projection(marker, raw_data, metadata), axis=1)
                except TypeError:
                    print(user, position, metadata)
                    raise
                marker_positions.append(marker_position)
            marker_positions = np.stack(marker_positions, axis=2)
            robot_pos = np.nanmean(marker_positions, axis=2)

            #participant's comfort during the approach (lower is better)
            comfort = get_comfort(raw_data, metadata)

            #participant's height
            height = participant_height(raw_data,metadata)
            if not type(height) is np.ndarray:
                # height is 0.0 because head markers couldn't be found
                height = np.array([np.nan] * robot_pos.shape[0])

            #human position
            human_pos = np.stack(marker_projection("UsersHead", raw_data, metadata), axis=1)

            # distance -- euclidean distance
            distance = get_distance(raw_data, metadata)

            #position and user label
            pos_label = np.array([positions.index(position)] * robot_pos.shape[0])
            user_label = np.array([user] * robot_pos.shape[0])

            #stack the data together
            data_rows = [
                        frames,
                        robot_pos[:,0], robot_pos[:,1],
                        distance, 
                        human_pos[:,0], human_pos[:,1], 
                        height, 
                        comfort, 
                        pos_label, 
                        user_label]
            partial_dataset = np.stack(data_rows, axis=1)
            dataset = np.append(dataset, partial_dataset, axis=0)
    return dataset

def save_dataset(dataset):
    dataset = generate_dataset()
    column_names = [
        "Frame",
        "RobotX", "RobotY",
        "distance",
        "HumanX", "HumanY",
        "HumanHeight",
        "comfort",
        "Angle",
        "ParticipantID"]
    np.savetxt("proximity_data.csv", dataset, delimiter=",", header=",".join(column_names))
    np.save("proximity_data.npy", dataset, fix_imports=False)
    np.save("proximity_data_no_compression.npy", dataset, allow_pickle=False)

    return dataset

if __name__ == "__main__":
    column_names = [
        "Frame",
        "RobX", "RobY",
        "distance",
        "HumX", "HumY",
        "HumZ",
        "comfort",
        "Angle",
        "ID"]

    dataset = generate_dataset()
    #save_dataset(dataset)
    #dataset = np.load("proximity_data.npy")
    dataset = dataset[dataset[:,-2]!=3]
    dataset = dataset[dataset[:,-2]!=4]

    wanted_columns = [
        column_names.index("distance"),
        column_names.index("Angle"),
        column_names.index("ID")
    ]
    dataset = dataset[:,wanted_columns]
    valid_rows = np.logical_not(np.any(np.isnan(dataset), axis=1))
    dataset = dataset[valid_rows,:]
    labels = np.ones(dataset.shape[0])
    new_dataset = np.empty((dataset.shape[0],dataset.shape[1]+1))
    new_dataset[:,:-1] = dataset
    new_dataset[:,-1] = labels
    dataset = new_dataset
    save_dataset(dataset)

    nans = np.sum(np.isnan(dataset),axis=0)
    missing = 1 - nans/dataset.shape[0]
    print(missing)
