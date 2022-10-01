from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random

class Emotion_DataLoader(Dataset):
    def __init__(self,csv,features_path,max_frames, feature_framerate, video_dim=1024):
        self.csv = pd.read_csv(csv)
        self.features_path = features_path
        self.max_frames = max_frames
        self.feature_size = video_dim
        self.feature_framerate = feature_framerate
        self.iter_num = len(self.csv)

    def __len__(self) -> int:
        return self.iter_num

    def __getitem__(self, feature_idx):
        idx = feature_idx

        video, video_mask, masked_video, video_labels_index = self._get_video(idx, np.array(0), np.array(1))
        audio, emotion_label = self._get_audio(idx)

        return audio,  video, video_mask, masked_video, video_labels_index, emotion_label

    def _expand_video_slice(self,s,e,si,ei,fps, video_features):
        start =0
        end = len(video_features)
        if start > end:
            start, end = end, start
        video_slice = video_features[start:end]

        expand_left = True
        #应该不会执行该while
        while len(video_slice) < 1:
            if si == 0 and ei == len(s) - 1:
                break
            if expand_left:
                expand_left = False
                si = si - 1 if si > 0 else si
            else:
                expand_left = True
                ei = ei + 1 if ei < len(e) - 1 else ei
            start = int(s[si] * fps)
            end = int(e[ei] * fps) + 1
            if start > end:
                start, end = end, start
            video_slice = video_features[start:end]

        if self.max_frames < video_slice.shape[0]:
            video_slice = video_slice[:self.max_frames]

        return video_slice, start, end


    def _get_video(self,idx,s,e):

        feature_file = os.path.join(self.features_path, self.csv["video_feature"].values[idx])
        video_features = np.load(feature_file)
        video_length = 1
        video_mask = np.zeros((video_length, self.max_frames), dtype=np.long)
        max_video_length = [0] * video_length
        video = np.zeros((video_length, self.max_frames, self.feature_size), dtype=np.float)
        for i in range(video_length):
            if len(video_features) < 1:
                raise ValueError("{} is empty.".format(feature_file))
            video_slice, start, end = self._expand_video_slice(s, e, i, i, self.feature_framerate, video_features)
            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                pass
            else:
                video[i][:slice_shape[0]] = video_slice
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(s))]
        masked_video = video.copy()
        # if only_sim is False:
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def _get_audio(self, idx):
        feature_file = os.path.join(self.features_path, self.csv["audio_feature"].values[idx])
        audio_features = np.load(feature_file)
        emotion_label = os.path.join(self.features_path, self.csv["emotion_label"].values[idx])
        return audio_features, emotion_label
