from enum import IntEnum


class MarkerColumn(IntEnum):
    COL_F_POS = 0,
    COL_PID = 1,
    COL_MID = 2,
    COL_U = 3,
    COL_V = 4,
    COL_C = 5,
    COL_ARRAY_ID = 6


def filter_markers_by_frame(the_markers, frame_pos):
    return the_markers[the_markers[:, MarkerColumn.COL_F_POS] == frame_pos]


def filter_markers_by_location(the_markers, lower_bound=0, upper_bound=1):
    return the_markers[(the_markers[:, MarkerColumn.COL_U] > lower_bound) &
                       (the_markers[:, MarkerColumn.COL_U] <= upper_bound) &
                       (the_markers[:, MarkerColumn.COL_V] > lower_bound) &
                       (the_markers[:, MarkerColumn.COL_V] <= upper_bound)]


def filter_markers_by_mids(the_markers, the_mid):
    if isinstance(the_mid, list):
        if len(the_mid) == 1:
            return the_markers[the_markers[:, MarkerColumn.COL_MID] == the_mid[0]]
        else:
            the_filter = []
            for mid in the_mid:
                if len(the_filter) == 0:
                    the_filter = the_markers[:, MarkerColumn.COL_MID] == mid
                else:
                    the_filter = (the_markers[:, MarkerColumn.COL_MID] == mid) | the_filter
            return the_markers[the_filter]

    return the_markers[the_markers[:, MarkerColumn.COL_MID] == the_mid]


def scale_and_round(marker, size):
    return (marker * size).astype(int)


def unpack_marker(marker):
    return marker[MarkerColumn.COL_F_POS],\
           marker[MarkerColumn.COL_PID],\
           marker[MarkerColumn.COL_MID],\
           marker[MarkerColumn.COL_U:MarkerColumn.COL_C],\
           marker[MarkerColumn.COL_C],\
           marker[MarkerColumn.COL_ARRAY_ID]
