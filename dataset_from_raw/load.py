from pprint import pprint
import numpy as np
import glob
import re

class CorruptedHeaderException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)

class CorruptedDataLineException(Exception):
    pass

def load(tsv_file):
    with open(tsv_file, "r") as input_stream:
        header = list()
        for idx, data_row in enumerate(input_stream):
            if idx > 9:
                break
            header.append(data_row.split("\t"))
        data_names = data_row.split("\t")[:-1]

        # parse header into a dict
        try:
            metadata = {
                "NO_OF_FRAMES":int(header[0][1]),
                "NO_OF_CAMERAS":int(header[1][1]),
                "NO_OF_MARKERS":int(header[2][1]),
                "FREQUENCY":int(header[3][1]),
                "NO_OF_ANALOG":int(header[4][1]),
                "ANALOG_FREQUENCY":int(header[5][1]),
                "DESCRIPTION":header[6][1],
                "TIME_STAMP":(header[7][1],float(header[7][2])),
                "DATA_INCLUDED":header[8][1],
                "MARKER_NAMES":header[9][1:],
                "COLUMN_NAMES":data_names
            }
        except IndexError:
            print("Corrupted Header at %s" % tsv_file)
            raise CorruptedHeaderException

        # parse data into a numpy array
        data = np.empty((metadata["NO_OF_FRAMES"],len(data_names)))
        for idx, data_row in enumerate(input_stream):
            data_row = data_row.split("\t")
            frame = int(data_row[0])
            try:
                data[idx,:] = [float(x) for x in data_row]
            except IndexError:
                print("Corrupted data for frame: %s" %frame)
                raise CorruptedDataLineException
    
    return data, metadata

def test_skip_frames(data):
    """Test if there are skipped frames. @returns True if all frames are there and in order @returns false otherwise"""
    first_frame = data[0,0]
    frames = data[:,0]
    frame_idx = frames - first_frame
    frame_valid = frame_idx == range(frame_idx.shape[0])
    if all(frame_valid):
        return True
    else:
        print("Some frames were skipped.")
        return False

def test_skip_data(data, metadata, column=0):
    if np.any(data[:,column] < 0.00001):
        print("Column %d: Emtpy fields found in column %s." % (column, metadata["COLUMN_NAMES"][column]))
        return False
    else:
        return True

def load_all(path):
    id_extractor = re.compile("User(\d+)_tr([A-E])")
    valid_positions = ["A","B","C","D","E"]
    by_position = {
        "A":list(),
        "B":list(),
        "C":list(),
        "D":list(),
        "E":list()
    }

    by_id = list()
    for _ in range(32):
        by_id.append(dict())

    for file_name in glob.iglob(path):
        result = re.search(id_extractor, file_name)
        if result:
            user_id = int(result.group(1)) - 1
            position = result.group(2)
            data, metadata = load(file_name)
            by_position[position].append({"data":data,"metadata":metadata})
            by_id[user_id][position] = dict()
            by_id[user_id][position]["metadata"] = metadata
            by_id[user_id][position]["data"] = data
        else:
            print("Could not extract user ID and position for %s" % file_name)

    # remove all users that don't have any label
    by_id = [element for element in by_id if element]
    return by_id, by_position

if __name__ == "__main__":
    by_id, by_position = load_all("../tsv_files/*.tsv")
    """
    for column in range(len(metadata["COLUMN_NAMES"])):
        intermediate_result = test_skip_data(data, metadata, column)
        test_result = test_result and intermediate_result
    if test_result:
        print("All tests passed. Data looks valid.")
    else:
        print("Some tests have failed, check log.")
    """
        
