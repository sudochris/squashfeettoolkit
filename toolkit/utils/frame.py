import json as json

class FrameEvents:

    def __init__(self, events_file) -> None:
        super().__init__()
        self.events_file = events_file
        """self.events = {
            "game_state":  {
                0: "rally_end",
                1: "rally_start"
            }
        }"""
        self.events = {}

        with open(events_file, 'r') as file:
            data = json.load(file)
            events_map = data["eventObjects"]
            for entry in events_map:
                frame = entry["framePosition"]
                for evt in entry["events"]:
                    group = evt["group"]
                    self.events.setdefault(group, {}).update({
                        frame: evt["name"]
                    })

    def last_group_event_change_for(self, frame_pos, group):
        for i in range(frame_pos, -1, -1):
            if i in self.events[group]:
                return self.events[group][i]
        return None