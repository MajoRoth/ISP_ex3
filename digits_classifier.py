from abc import abstractmethod
import torch
import typing as tp
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import euclidian


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    # we will use this to give an absolute path to the data, make sure you read the data using this argument. 
    # you may assume the train data is the same
    path_to_training_data_dir: str = "./train_files"

    # you may add other args here


class DigitClassifier():
    """
    You should Implement your classifier object here
    """

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir

    @abstractmethod
    def classify_using_eucledian_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        raise NotImplementedError("function is not implemented")

    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """
        raise NotImplementedError("function is not implemented")

    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """
        raise NotImplementedError("function is not implemented")

    def DTW_distance(self, mfcc_1: np.ndarray, mfcc_2: np.ndarray):

        n = len(mfcc_1)
        m = len(mfcc_2)
        distance_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                distance_matrix[i, j] = euclidian(mfcc_1[i],
                                                  mfcc_2[j])  # TODO check if the dimensions of the mfcc's makes sense

        # find optimal path
        DTW_path = []
        i, j = (0, 0)
        while i < n - 1 and j < m - 1:
            neighbors = [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
            distances = [distance_matrix[p] for p in neighbors]
            minimal_step_index = np.argmin(distances)
            i, j = neighbors[minimal_step_index]
            DTW_path.append((i, j))

        return sum([distance_matrix[p] for p in DTW_path])


class ClassifierHandler:

    @staticmethod
    def get_pretrained_model() -> DigitClassifier:
        """
        This function should load a pretrained / tuned 'DigitClassifier' object.
        We will use this object to evaluate your classifications
        """
        raise NotImplementedError("function is not implemented")
