import os
from abc import abstractmethod
import torch
import typing as tp
from dataclasses import dataclass
import numpy as np
import librosa
from scipy.spatial.distance import euclidian

numbers_translate = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

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
        self.train_data_dict = dict()  # dict with 1,2,3,4,5 as keys, and values are lists with mfccs

        train_files = os.listdir(self.path_to_training_data)
        for number_dir in train_files:
            number_dir_path = os.path.join(self.path_to_training_data, number_dir)
            if os.path.isdir(number_dir_path):
                self.train_data_dict[numbers_translate[number_dir]] = list()
                wavs = os.listdir(number_dir_path)
                for wav in wavs:
                    if wav.endswith(".wav"):
                        wav_path = os.path.join(number_dir_path, wav)
                        self.train_data_dict[numbers_translate[number_dir]].append(self.get_mfcc(wav_path))

        #  print(self.train_data_dict)

    def get_mfcc(self, wav_path: tp.Union[tp.List[str], str]):
        """
        gets a single str or list of str and returns the mfccs
        :param wav_path:
        :return:
        """
        if type(wav_path) is list:
            mfcc_list = list()
            for wav in wav_path:
                y, sr = librosa.load(wav, sr=None)
                mfcc_list.append(librosa.feature.mfcc(y=y, sr=sr))
            return mfcc_list

        y, sr = librosa.load(wav_path, sr=None)
        return librosa.feature.mfcc(y=y, sr=sr)



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

        return_list = list()

        dtw_results = self.classify_using_DTW_distance(self.get_mfcc(audio_files))
        eucledian_results = self.classify_using_DTW_distance(self.get_mfcc(audio_files))

        for i, audio_file in enumerate(audio_files):
            return_list.append(
                f"{audio_file} - {dtw_results[i]} - {eucledian_results[i]}"
            )

        return return_list


    def DTW_distance(self, mfcc_1: np.ndarray, mfcc_2: np.ndarray):
        # todo - i didn't check if this function works correctly

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


if __name__ == '__main__':
    ca = ClassifierArgs()
    ca.path_to_training_data_dir = "./train_files"
    dc = DigitClassifier(ca)