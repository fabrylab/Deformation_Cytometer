from deformationcytometer.detection.includes.pipe_helpers import *


class ProcessPairData:
    def __call__(self, data):
        from deformationcytometer.detection.includes.regionprops import matchVelocities

        if data["type"] == "end" or data["type"] == "start":
            return data

        log("4vel", "prepare", 1, data["index"])

        cells = data["cells"]
        row_indices = data["row_indices"]
        next_id = 1+data["index"]*1000

        _, next_id = matchVelocities(cells.iloc[0:0],
                                     cells.iloc[row_indices[0]:row_indices[1]],
                                     next_id, data["config"])

        for i, index in enumerate(range(data["index"], data["end_index"]-1)):
            _, next_id = matchVelocities(cells.iloc[row_indices[i]:row_indices[i+1]],
                                         cells.iloc[row_indices[i+1]:row_indices[i+2]],
                                         next_id, data["config"])

        log("4vel", "prepare", 0, data["index"])
        return data
