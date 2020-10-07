from enum import IntEnum


class BodyParts(IntEnum):
    pass


class OpenPoseBody25(BodyParts):
    Nose = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    MidHip = 8,
    RHip = 9,
    RKnee = 10,
    RAnkle = 11,
    LHip = 12,
    LKnee = 13,
    LAnkle = 14,
    REye = 15,
    LEye = 16,
    REar = 17,
    LEar = 18,
    LBigToe = 19,
    LSmallToe = 20,
    LHeel = 21,
    RBigToe = 22,
    RSmallToe = 23,
    RHeel = 24,
    Background = 25


class OpenPoseCoco(BodyParts):
    Nose = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    RHip = 8,
    RKnee = 9,
    RAnkle = 10,
    LHip = 11,
    LKnee = 12,
    LAnkle = 13,
    REye = 14,
    LEye = 15,
    REar = 16,
    LEar = 17,
    Background = 18


class OpenPoseMpi(BodyParts):
    Head = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    RHip = 8,
    RKnee = 9,
    RAnkle = 10,
    LHip = 11,
    LKnee = 12,
    LAnkle = 13,
    Chest = 14,
    Background = 15


class PoseNet(BodyParts):
    Nose = 0,
    LEye = 1,
    REye = 2,
    LEar = 3,
    REar = 4,
    LShoulder = 5,
    RShoulder = 6,
    LElbow = 7,
    RElbow = 8,
    LWrist = 9,
    RWrist = 10,
    LHip = 11,
    RHip = 12,
    LKnee = 13,
    RKnee = 14,
    LAnkle = 15,
    RAnkle = 16


class Annotation(BodyParts):
    LFootP1 = 0,
    RFootP1 = 1,
    LFootP2 = 2,
    RFootP2 = 3


class ArtTrack(BodyParts):
    Nose = 0,
    LEye = 1,
    REye = 2,
    LEar = 3,
    REar = 4,
    LShoulder = 5,
    RShoulder = 6,
    LElbow = 7,
    RElbow = 8,
    LWrist = 9,
    RWrist = 10,
    LHip = 11,
    RHip = 12,
    LKnee = 13,
    RKnee = 14,
    LAnkle = 15,
    RAnkle = 16