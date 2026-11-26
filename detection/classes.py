"""Shared class definitions for the whole team.

OUR_CLASSES        -> the 6 labels used everywhere (generator GT, events CSV, dashboard, eval)
DATASET_TO_OURS    -> which ESC-50 / UrbanSound8K labels Person A mixes in as each class
YAMNET_TO_OURS     -> which of YAMNet's 521 AudioSet classes count as each of our classes
DEFAULT_THRESHOLDS -> onset thresholds per class (tune with quick_check.py --sweep)

All YAMNet names below were checked against yamnet_class_map.csv (display_name column).
"""

OUR_CLASSES = ["traffic", "voices", "alarm", "dog_bark", "snoring", "door"]

# Source labels Person A should use for each of our classes.
# Team decision (Nov 2026): car_horn -> alarm, door = knocks only.
# Note: neither dataset has plain adult speech, so "voices" = laughing, crying baby, children playing.
DATASET_TO_OURS = {
    "esc50": {
        "car_horn": "alarm",          # YAMNet hears horns as Beep/Car alarm -> grouped with alarm
        "engine": "traffic",
        "laughing": "voices",
        "crying_baby": "voices",
        "siren": "alarm",
        "clock_alarm": "alarm",
        "dog": "dog_bark",
        "snoring": "snoring",
        "door_wood_knock": "door",    # creaks dropped: YAMNet hears them as Buzzer/Snoring
    },
    "urbansound8k": {
        "car_horn": "alarm",
        "engine_idling": "traffic",
        "children_playing": "voices",
        "siren": "alarm",
        "dog_bark": "dog_bark",
    },
}

YAMNET_TO_OURS = {
    "traffic": [
        "Vehicle", "Motor vehicle (road)", "Car", "Car passing by",
        "Traffic noise, roadway noise", "Truck", "Bus", "Motorcycle",
        "Engine", "Idling", "Engine starting", "Accelerating, revving, vroom",
        "Light engine (high frequency)", "Medium engine (mid frequency)",
        "Heavy engine (low frequency)", "Tire squeal", "Skidding",
    ],
    "voices": [
        "Speech", "Child speech, kid speaking", "Conversation", "Babbling",
        "Shout", "Yell", "Children shouting", "Screaming", "Whispering",
        "Laughter", "Baby laughter", "Belly laugh", "Crying, sobbing",
        "Baby cry, infant cry", "Chatter", "Hubbub, speech noise, speech babble",
        "Children playing",
    ],
    "alarm": [
        "Alarm", "Alarm clock", "Siren", "Civil defense siren", "Emergency vehicle",
        "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
        "Buzzer", "Smoke detector, smoke alarm", "Fire alarm", "Car alarm",
        "Beep, bleep", "Telephone bell ringing", "Ringtone",
        "Vehicle horn, car horn, honking", "Air horn, truck horn", "Toot",
    ],
    "dog_bark": [
        "Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)",
        "Canidae, dogs, wolves",
    ],
    "snoring": ["Snoring", "Snort"],
    "door": ["Door", "Knock", "Slam", "Sliding door", "Cupboard open or close"],
}

# Starting points only. Snoring/door score lower in YAMNet, so they start lower.
DEFAULT_THRESHOLDS = {
    "traffic": 0.20,
    "voices": 0.20,
    "alarm": 0.20,
    "dog_bark": 0.20,
    "snoring": 0.15,
    "door": 0.15,
}
