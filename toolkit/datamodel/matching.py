
from enum import IntEnum

from toolkit.datamodel.bodyparts import ArtTrack, Annotation, OpenPoseBody25, OpenPoseCoco, OpenPoseMpi, PoseNet
from toolkit.utils.configuration import AlgorithmType

class FrameExperiments(IntEnum):
    """FA, FR, FN"""
    ALL_FRAMES = 0,
    RALLY_ONLY = 1,
    NONRALLY_ONLY = 2


class Experiments(IntEnum):
    """MA, MI"""
    MATCH_ALL = 0,
    MATCH_INDIVIDUAL = 1

    @staticmethod
    def generate_match_all():
        """MA matching type"""
        return {
            AlgorithmType.A0_ARTTRACK: {
                ArtTrack.LAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2],
                ArtTrack.RAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A1F0_OPENPOSE_BODY25: {
                OpenPoseBody25.LHeel: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2],
                OpenPoseBody25.RHeel: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A1F1_OPENPOSE_COCO: {
                OpenPoseCoco.LAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2],
                OpenPoseCoco.RAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A1F2_OPENPOSE_MPI: {
                OpenPoseMpi.LAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2],
                OpenPoseMpi.RAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A2_POSENET: {
                PoseNet.LAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2],
                PoseNet.RAnkle: [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]
            }
        }

    @staticmethod
    def generate_match_individual():
        """MI matching type"""
        return {
            AlgorithmType.A0_ARTTRACK: {
                ArtTrack.LAnkle: [Annotation.LFootP1, Annotation.LFootP2],
                ArtTrack.RAnkle: [Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A1F0_OPENPOSE_BODY25: {
                OpenPoseBody25.LHeel: [Annotation.LFootP1, Annotation.LFootP2],
                OpenPoseBody25.RHeel: [Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A1F1_OPENPOSE_COCO: {
                OpenPoseCoco.LAnkle: [Annotation.LFootP1, Annotation.LFootP2],
                OpenPoseCoco.RAnkle: [Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A1F2_OPENPOSE_MPI: {
                OpenPoseMpi.LAnkle: [Annotation.LFootP1, Annotation.LFootP2],
                OpenPoseMpi.RAnkle: [Annotation.RFootP1, Annotation.RFootP2]
            },
            AlgorithmType.A2_POSENET: {
                PoseNet.LAnkle: [Annotation.LFootP1, Annotation.LFootP2],
                PoseNet.RAnkle: [Annotation.RFootP1, Annotation.RFootP2]
            }
        }

def get_matcher_for_experiment(experiment : Experiments):
    if experiment == Experiments.MATCH_ALL:
        return Experiments.generate_match_all()
    elif experiment == Experiments.MATCH_INDIVIDUAL:
        return Experiments.generate_match_individual()