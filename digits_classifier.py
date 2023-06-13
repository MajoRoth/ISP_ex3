import os
from abc import abstractmethod
import torch
import typing as tp
from dataclasses import dataclass
import numpy as np
import librosa
from scipy.spatial import distance
import torchaudio
from dtw import dtw

numbers_translate = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}


def list_test_files(test_set_path):
    test_files_list = []
    basedir = os.path.abspath(test_set_path)
    for file in os.listdir(basedir):
        if not file.endswith(".wav"):
            continue
        fullpath = os.path.join(basedir, file)
        test_files_list.append(fullpath)

    return test_files_list


######### TODO - tmp #######################
import pandas as pd
def evaluate(euclidean_results, dtw_results):
    # evaluation
    test_GT_labels_df = pd.read_csv("test_labels.csv")

    # parse GT labels
    GT_labels = []
    GT_filelist = list(test_GT_labels_df['filename'])
    for tf in test_list:
        idx = GT_filelist.index(tf.split("/")[-1])
        GT_labels.append(test_GT_labels_df['label'][idx])

    # evaluate
    GT_labels = np.array(GT_labels)
    euclidean_preds = np.array(euclidean_results)
    dtw_preds = np.array(dtw_results)

    valid_GT_idxs = GT_labels > -1
    euc_accuracy = np.mean(euclidean_preds[valid_GT_idxs] == GT_labels[valid_GT_idxs])
    dtw_accuracy = np.mean(dtw_preds[valid_GT_idxs] == GT_labels[valid_GT_idxs])

    print("classify")
    print(f'euc accuracy: {euc_accuracy}')
    print(f'dtw accuracy: {dtw_accuracy}')
###########################################################

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

    class TrainData():
        def __init__(self):
            self.features = None # mfcc features - [Batch, Channels, n_mfccs, n_frames]
            self.labels = None # 1,2,3,4,5 digits

    def __init__(self, args: ClassifierArgs):
        self.path_to_training_data = args.path_to_training_data_dir
        self.sr = None
        self.calc_train_features()

    def load_audio_files(self, audio_files: tp.List[str]):
        y_all = None
        for audio_f in audio_files:
            y, sr = torchaudio.load(audio_f)
            if y_all is None:
                y_all = y.unsqueeze(0)
            else:
                y_all = torch.cat((y_all, y.unsqueeze(0)), dim=0)
        return y_all, sr # [Batch, Channels, Time]

    # def get_mfcc(self, wav_path: tp.Union[tp.List[str], str]):
    #     """
    #     gets a single str or list of str and returns the mfccs
    #     :param wav_path:
    #     :return:
    #     """
    #     if type(wav_path) is list:
    #         mfcc_list = list()
    #         for wav in wav_path:
    #             y, sr = librosa.load(wav, sr=None)
    #             assert sr == self.sr
    #             mfcc_list.append(librosa.feature.mfcc(y=y, sr=sr))
    #         return mfcc_list
    #
    #     y, sr = librosa.load(wav_path, sr=None)
    #     assert sr == self.sr
    #     return librosa.feature.mfcc(y=y, sr=sr), sr

    def calc_train_features(self):
        self.train_ds = self.TrainData()

        train_subdirs = os.listdir(self.path_to_training_data)


        for number_dir in train_subdirs:
            number_dir_path = os.path.join(self.path_to_training_data, number_dir)
            if os.path.isdir(number_dir_path):
                wavs = os.listdir(number_dir_path)
                for wav in wavs:
                    if not wav.endswith(".wav"):
                        continue
                    wav_path = os.path.join(number_dir_path, wav)
                    y, sr = torchaudio.load(wav_path)
                    if self.sr is None:
                        self.sr = sr
                    features = librosa.feature.mfcc(y=y.numpy(), sr=sr)
                    label = numbers_translate[number_dir]
                    if self.train_ds.features is None:
                        self.train_ds.features = np.expand_dims(features, 0)
                        self.train_ds.labels = [label]
                    else:
                        self.train_ds.features = np.concatenate((self.train_ds.features, np.expand_dims(features, 0)), axis=0)
                        self.train_ds.labels.append(label)

        # convert to torch.Tensor
        if self.train_ds.features is not None:
            self.train_ds.features = torch.from_numpy(self.train_ds.features)
            self.train_ds.labels = torch.tensor(self.train_ds.labels)


    @abstractmethod
    def classify_using_euclidean_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """

        # calc test features
        x_mfccs = librosa.feature.mfcc(y=audio_files.numpy(), sr=self.sr) # [Batch, Channels, n_mfccs, n_frames]
        x_mfccs = torch.from_numpy(x_mfccs)

        # rearrange
        n_test_samples, ch, n_mfccs, n_frames = x_mfccs.shape
        x_mfccs = x_mfccs.permute(1, 0, 2, 3)

        n_train_samples, ch_, n_mfccs_, n_frames_ = self.train_ds.features.shape
        assert ch_ == ch and n_mfccs_ == n_mfccs_ and n_frames_ == n_frames

        train_feats = self.train_ds.features.permute(1, 0, 2, 3)

        # for each temporal mfcc frame - calc pairwise distances
        all_pair_wise_dists = None
        for t in range(n_frames):
            pair_wise_dists = torch.cdist(x_mfccs[:, :, :, t], train_feats[:, :, :, t])
            if all_pair_wise_dists is None:
                all_pair_wise_dists = pair_wise_dists.unsqueeze(0)
            else:
                all_pair_wise_dists = torch.cat((all_pair_wise_dists, pair_wise_dists.unsqueeze(0)), dim=0)

        # calc mean dist for each test-train pair (over the temporal dimension)
        mean_dists = all_pair_wise_dists.mean(dim=0).mean(dim=0) # mean over temporal dim and over audio channels

        # mean_dists dim is now [n_test_examples, n_train_examples]
        # For each test example, classify by the minimal distance from train examples
        nn_idx = torch.argmin(mean_dists, dim=1)
        pred_labels = self.train_ds.labels[nn_idx]

        return pred_labels.tolist()


    @abstractmethod
    def classify_using_DTW_distance(self, audio_files: tp.Union[tp.List[str], torch.Tensor]) -> tp.List[int]:
        """
        function to classify a given audio using DTW distance
        audio_files: list of audio file paths or a a batch of audio files of shape [Batch, Channels, Time]
        return: list of predicted label for each batch entry
        """

        # calc test features
        x_mfccs = librosa.feature.mfcc(y=audio_files.numpy(), sr=self.sr)  # [Batch, Channels, n_mfccs, n_frames]

        # rearrange
        n_test_samples, ch, n_mfccs, n_frames = x_mfccs.shape

        n_train_samples, ch_, n_mfccs_, n_frames_ = self.train_ds.features.shape

        assert ch_ == ch and n_mfccs_ == n_mfccs_ and n_frames_ == n_frames


        # for each temporal mfcc frame - calc pairwise distances
        all_pair_wise_dists = DigitClassifier.DTWdist(x_mfccs, self.train_ds.features.numpy())

        # calc mean dist for each test-train pair (over the temporal dimension)

        # mean_dists dim is now [n_test_examples, n_train_examples]
        # For each test example, classify by the minimal distance from train examples
        nn_idx = torch.argmin(torch.from_numpy(all_pair_wise_dists), dim=1)
        pred_labels = self.train_ds.labels[nn_idx]


        return pred_labels.tolist()



    @abstractmethod
    def classify(self, audio_files: tp.List[str]) -> tp.List[str]:
        """
        function to classify a given audio using auclidean distance
        audio_files: list of ABSOLUTE audio file paths
        return: a list of strings of the following format: '{filename} - {predict using euclidean distance} - {predict using DTW distance}'
        Note: filename should not include parent path, but only the file name itself.
        """

        return_list = list()
        x, sr = self.load_audio_files(audio_files)
        euclidean_results = self.classify_using_euclidean_distance(x)

        DTW_IS_IMPLEMENTED = True
        if DTW_IS_IMPLEMENTED:
            dtw_results = self.classify_using_DTW_distance(x)
        else:
            dtw_results = euclidean_results

        for i, audio_file in enumerate(audio_files):
            return_list.append(
                f"{audio_file} - {dtw_results[i]} - {euclidean_results[i]}"
            )

        ### TODO - TMP #####
        evaluate(euclidean_results, dtw_results)
        ####################

        return return_list

    @staticmethod
    def DTWdist(batch_1, batch_2):
        """
        Calculating batched DTW metric
        :param x1:
        :param x2:
        :return:
        """

        n = batch_1.shape[0]
        m = batch_2.shape[0]

        return_dist = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                return_dist[i, j], _, _, _ = dtw(batch_1[i], batch_2[j], dist=lambda x, y: np.linalg.norm(x - y, ord=1))

        return return_dist


    # def DTW_distance(self, mfcc_1: np.ndarray, mfcc_2: np.ndarray):
    #     # todo - i didn't check if this function works correctly
    #
    #     n = len(mfcc_1)
    #     m = len(mfcc_2)
    #     distance_matrix = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             distance_matrix[i, j] = distance.euclidian(mfcc_1[i],
    #                                               mfcc_2[j])  # TODO check if the dimensions of the mfcc's makes sense
    #
    #     # find optimal path
    #     DTW_path = []
    #     i, j = (0, 0)
    #     while i < n - 1 and j < m - 1:
    #         neighbors = [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
    #         distances = [distance_matrix[p] for p in neighbors]
    #         minimal_step_index = np.argmin(distances)
    #         i, j = neighbors[minimal_step_index]
    #         DTW_path.append((i, j))
    #
    #     return sum([distance_matrix[p] for p in DTW_path])


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
    test_list = list_test_files("./test_files")
    classification_res = dc.classify(test_list)
    print('completed')