import numpy as np
import pandas as pd

class FrameResult:
    """
    frame_pos (row_idx), [tpi, fpi, fni]{num_threshs}
    """
    def __init__(self, num_frames, num_threshs) -> None:
        super().__init__()
        n_rows = num_frames
        self.col_names = []
        for i in range(num_threshs):
            self.col_names += [f"TP{i}", f"FP{i}", f"FN{i}"]

        self.col_names.append("frame_is_rally")
        n_cols = len(self.col_names)

        self.result_matrix = np.zeros((n_rows, n_cols))

    def set_is_rally_for_frame(self, frame_pos, is_rally):
        self.result_matrix[frame_pos, -1] = 1 if is_rally else 0
        pass

    def set_data_for(self, frame_pos, num_thresh, tp, fp, fn):
        sub_start = (num_thresh*3)
        sub_end = sub_start + 3
        self.result_matrix[frame_pos, sub_start:sub_end] = [tp, fp, fn]

    def save(self, csv_filename):
        pandas_result = pd.DataFrame(data=self.result_matrix, columns=self.col_names)
        pandas_result = pandas_result[(pandas_result.T != 0).any()]
        pandas_result.to_csv(csv_filename)

class MarkerResult:
    def __init__(self) -> None:
        super().__init__()
        self.col_names = ["frame_pos", "thresh_num", "mid", "tp", "fp", "px_err", "norm_err", "confidence", "frame_is_rally"]
        self.result_matrix = []
        self.row_idx = 0

    def add_data_for(self, frame_pos, thresh_num, mid, tp, px_err, norm_err, confidence, frame_is_rally):
        self.result_matrix.append([frame_pos, thresh_num, mid, 1 if tp else 0, 0 if tp else 1, px_err, norm_err, confidence, 1 if frame_is_rally else 0])
        self.row_idx += 1

    def save(self, csv_filename):
        pandas_result = pd.DataFrame(data=self.result_matrix, columns=self.col_names)
        pandas_result.to_csv(csv_filename)
