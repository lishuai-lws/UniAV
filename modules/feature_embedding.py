#  导入模块许需要统一加入的，还没找到更好的解决办法
import csv
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from modules.module_embedding import AudioWav2Vec2, ResNet50
import argparse
import librosa
import os
import pandas as pd
import numpy as np
from PIL import Image
import math
# import cv2
import random
import json
from transformers import logging
import torch
from tqdm import tqdm

# warning level
logging.set_verbosity_error()



def get_args(description='data embedding'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--wav2vec2_base_960h",
                        default="/public/home/zwchen209/lishuai/pretrainedmodel/wav2vec2-base-960h",
                        help="pretrained wav2vect2.0 path")
    parser.add_argument("--resnet50", default="/home/lishuai/pretrainedmodel/resnet-50",
                        help="pretrained resnet50 path")
    parser.add_argument("--cmumosei_ids_path", default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/ids.csv")
    parser.add_argument("--cmumosei_audio_path",
                        default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/audio/WAV_fromVideo")
    parser.add_argument("--cmumosei_video_path",
                        default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/video/version_img_size_224_img_scale_1.3")
    parser.add_argument("--csv_path", default="/home/lishuai/workspace/data/cmumosei.csv")
    parser.add_argument("--feature_path", default="/home/lishuai/workspace/feature/cmumosei")
    parser.add_argument("--unlabeled_feature_path", default="/public/home/zwchen209/lishuai/feature")
    parser.add_argument("--max_seq_length", default=512, help=" max sequence length for encoder")
    parser.add_argument("--audiofeature_persecond", default=50, help=" wav2vec2.0 per second feature numbers")
    parser.add_argument("--min_audio_second", default=1, help=" min second of audio")
    parser.add_argument("--unlabeled_data_path", default="/public/home/zwchen209/Face-Detect-Track-Extract-main/output")
    parser.add_argument("--unlabeled_data_csv_path", default="/public/home/zwchen209/lishuai/data/unlabeled_data.csv")
    parser.add_argument("--loaded_csv_path", default="/public/home/zwchen209/lishuai/output/loaded_csv.json")
    parser.add_argument("--embedded_audio_json_path",
                        default="/public/home/zwchen209/lishuai/output/embedded_audio.json")
    parser.add_argument("--embedded_visual_json_path",
                        default="/public/home/zwchen209/lishuai/output/embedded_visual.json")

    args = parser.parse_args()

    return args


# 读取json文件到字典

def load_json(filename: str) -> dict:
    file = open(filename, 'r', encoding='UTF-8')
    js = file.read()
    file.close()

    dic = json.loads(js)
    return dic


# 从字典到json
def save_json(data: dict, filename: str) -> None:
    js_content = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False)
    file = open(filename, 'w', encoding='UTF-8')
    file.write(js_content)
    file.close()


def unlabeled_data_csv(opts):
    data_path = opts.unlabeled_data_path
    csv_path = opts.unlabeled_data_csv_path
    audios_path = os.path.join(data_path, "audio")
    videos_path = os.path.join(data_path, "images")
    files_list = os.listdir(audios_path)
    df = pd.DataFrame(columns=["audio_feature", "video_feature"])
    df.to_csv(csv_path, index=False)
    check_json = load_json(opts.loaded_csv_path)

    for file in tqdm(files_list):
        if file in check_json:
            print(file, "have loaded csv")
            continue
        file_path = os.path.join(audios_path, file)
        audios_list = os.listdir(file_path)
        for audio in audios_list:
            id = file + "_" + audio[:-4]
            # print("id:", id)
            audio_path = os.path.join(file_path, audio)
            # load audio wava
            audio_second = librosa.get_duration(filename=audio_path)
            max_audio_second = opts.max_seq_length / opts.audiofeature_persecond  # max audio length
            # audio length > max audio length
            if audio_second > max_audio_second:
                segment_num = math.ceil(audio_second / (max_audio_second - 1))
                for seg in range(segment_num):
                    # print ("seg:",seg)
                    # extract audio features
                    a_start = seg * (max_audio_second - 1)
                    a_end = min(audio_second, a_start + max_audio_second)
                    # print("start:",a_start,"end:",a_end) 
                    # Discard less than the shortest speech duration
                    if a_end - a_start < opts.min_audio_second:
                        continue

                    # audioFeature = audio_Wav2Vec2(opts,wave_data[a_start:a_end])
                    # print("audioFeature.shape:",audioFeature.shape)

                    # extract video feature
                    # video_length = len(video)
                    # v_start = math.floor(video_length*a_start/wave_data.shape[0])
                    # v_end = math.ceil(video_length*a_end/wave_data.shape[0])
                    # print("v_start:",v_start,"v_end:",v_end)
                    # videoFeature = video_resnet50(opts,video[v_start:v_end])
                    # print("len(videoFeature):",len(videoFeature))

                    # save feature to_csv
                    audio_file = "audio/" + id + "_" + str(seg) + ".npy"
                    video_file = "video/" + id + "_" + str(seg) + ".npy"
                    # audioFeaturePath = os.path.join(opts.feature_path,audio_file)
                    # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                    # np.save(audioFeaturePath,audioFeature)
                    # np.save(videoFeaturePath,videoFeature)

                    df = pd.DataFrame([[audio_file, video_file]])
                    df.to_csv(csv_path, mode="a", index=False, header=False)
                    del df
            else:
                # Discard less than the shortest speech duration
                if audio_second < opts.min_audio_second:
                    continue
                # extract audio features
                # audioFeature = audio_Wav2Vec2(opts,wave_data)
                # print(audioFeature.shape)

                # extract video feature
                # videoFeature = video_resnet50(opts,video)
                # print(len(videoFeature))

                # save feature to_csv
                audio_file = "audio/" + id + ".npy"
                video_file = "video/" + id + ".npy"
                # audioFeaturePath = os.path.join(opts.feature_path,audio_file)
                # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                # np.save(audioFeaturePath,audioFeature)
                # np.save(videoFeaturePath,videoFeature)

                # df = pd.DataFrame([[audio_file,video_file,label]])
                df = pd.DataFrame([[audio_file, video_file]])
                df.to_csv(csv_path, mode="a", index=False, header=False)
                del df
        check_json[file] = 1
        save_json(check_json, opts.loaded_csv_path)


def unlabeled_audio_data_embedding(opts):
    data_path = opts.unlabeled_data_path
    audios_path = os.path.join(data_path, "audio")
    videos_path = os.path.join(data_path, "images")
    feature_path = opts.unlabeled_feature_path
    files_list = os.listdir(audios_path)
    json_path = opts.embedded_audio_json_path
    check_json = load_json(json_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load modal
    modelpath = opts.wav2vec2_base_960h
    wav2vec_model = AudioWav2Vec2(modelpath, device)

    for file in tqdm(files_list):
        if file in check_json:
            print("audio file:", file, "have embedded")
            continue
        file_path = os.path.join(audios_path, file)
        audios_list = os.listdir(file_path)
        for audio in audios_list:
            id = file + "_" + audio[:-4]
            # print("id:", id)
            audio_path = os.path.join(file_path, audio)
            # load audio wava
            audio_second = librosa.get_duration(filename=audio_path)
            max_audio_second = opts.max_seq_length / opts.audiofeature_persecond  # max audio length
            wave_data, samplerate = librosa.load(audio_path, sr=16000)
            # audio length > max audio length
            if audio_second > max_audio_second:
                segment_num = math.ceil(audio_second / (max_audio_second - 1))
                for seg in range(segment_num):
                    # print ("seg:",seg)

                    start = seg * (max_audio_second - 1)
                    end = min(audio_second, start + max_audio_second)
                    # print("start:",a_start,"end:",a_end) 
                    # Discard less than the shortest speech duration
                    if end - start < opts.min_audio_second:
                        continue

                    # extract audio features

                    a_start = int(start * samplerate)
                    a_end = int(end * samplerate)
                    audioFeature = wav2vec_model(wave_data[a_start:a_end]).detach().cpu().numpy()

                    # print("audioFeature.shape:",audioFeature.shape)

                    # extract video feature
                    # video_length = len(video)
                    # v_start = math.floor(video_length*a_start/wave_data.shape[0])
                    # v_end = math.ceil(video_length*a_end/wave_data.shape[0])
                    # print("v_start:",v_start,"v_end:",v_end)
                    # videoFeature = video_resnet50(opts,video[v_start:v_end])
                    # print("len(videoFeature):",len(videoFeature))

                    # save feature to_csv
                    audio_file = "audio/" + id + "_" + str(seg) + ".npy"
                    video_file = "video/" + id + "_" + str(seg) + ".npy"
                    audioFeaturePath = os.path.join(feature_path, audio_file)
                    # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                    np.save(audioFeaturePath, audioFeature)
                    # np.save(videoFeaturePath,videoFeature)

                    # df = pd.DataFrame([[audio_file,video_file]])
                    # df.to_csv(csv_path,mode="a",index=False, header=False)
                    # del df 
            else:
                # Discard less than the shortest speech duration
                if audio_second < opts.min_audio_second:
                    continue
                # extract audio features
                # audioFeature = audio_Wav2Vec2(opts,wave_data)
                audioFeature = wav2vec_model(wave_data).detach().cpu().numpy()
                # print(audioFeature.shape)

                # extract video feature
                # videoFeature = video_resnet50(opts,video)
                # print(len(videoFeature))

                # save feature to_csv
                audio_file = "audio/" + id + ".npy"
                # video_file = "video/"+id+".npy"
                audioFeaturePath = os.path.join(feature_path, audio_file)
                # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                np.save(audioFeaturePath, audioFeature)
                # np.save(videoFeaturePath,videoFeature)

        check_json[file] = 1
        save_json(check_json, json_path)

def unlabeled_visual_data_embedding(opts):
    data_path = opts.unlabeled_data_path
    audios_path = os.path.join(data_path, "audio")
    videos_path = os.path.join(data_path, "images")
    feature_path = opts.unlabeled_feature_path

    files_list = os.listdir(audios_path)
    json_path = opts.embedded_visual_json_path
    check_json = load_json(json_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load modal
    modelpath = opts.resnet50
    resnet50_model = ResNet50(modelpath, device)

    for file in tqdm(files_list):
        if file in check_json:
            print("visual file:", file, "have embedded")
            continue
        file_path = os.path.join(audios_path, file)
        audios_list = os.listdir(file_path)
        video_file_path = os.path.join(videos_path, file)
        for audio in audios_list:
            id = file + "_" + audio[:-4]
            # print("id:", id)
            audio_path = os.path.join(file_path, audio)
            video_path = os.path.join(video_file_path, audio[:-4])
            # load audio wave
            audio_second = librosa.get_duration(filename=audio_path)
            max_audio_second = opts.max_seq_length / opts.audiofeature_persecond  # max audio length
            images_list = os.listdir(video_path)
            video = []
            for image in images_list:
                img = Image.open(os.path.join(video_path,image))
                video.append(np.array(img))
                img.close()
            # audio length > max audio length
            if audio_second > max_audio_second:
                segment_num = math.ceil(audio_second / (max_audio_second - 1))
                for seg in range(segment_num):
                    # print ("seg:",seg)

                    start = seg * (max_audio_second - 1)
                    end = min(audio_second, start + max_audio_second)
                    # print("start:",a_start,"end:",a_end)
                    # Discard less than the shortest speech duration
                    if end - start < opts.min_audio_second:
                        continue

                    # extract audio features

                    # a_start = int(start * samplerate)
                    # a_end = int(end * samplerate)
                    # audioFeature = wav2vec_model(wave_data[a_start:a_end]).detach().cpu().numpy()

                    # print("audioFeature.shape:",audioFeature.shape)

                    # extract video feature
                    video_length = len(video)
                    v_start = math.floor(video_length * start/audio_second)
                    v_end = math.ceil(video_length * end/audio_second)
                    print("v_start:",v_start,"v_end:",v_end)
                    # todo: for
                    videoFeature = resnet50_model(opts,video[v_start:v_end]).detach().cpu().numpy()
                    # print("len(videoFeature):",len(videoFeature))

                    # save feature to_csv
                    audio_file = "audio/" + id + "_" + str(seg) + ".npy"
                    video_file = "video/" + id + "_" + str(seg) + ".npy"
                    audioFeaturePath = os.path.join(feature_path, audio_file)
                    # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                    np.save(audioFeaturePath, audioFeature)
                    # np.save(videoFeaturePath,videoFeature)

                    # df = pd.DataFrame([[audio_file,video_file]])
                    # df.to_csv(csv_path,mode="a",index=False, header=False)
                    # del df
            else:
                # Discard less than the shortest speech duration
                if audio_second < opts.min_audio_second:
                    continue
                # extract audio features
                # audioFeature = audio_Wav2Vec2(opts,wave_data)
                audioFeature = wav2vec_model(wave_data).detach().cpu().numpy()
                # print(audioFeature.shape)

                # extract video feature
                # videoFeature = video_resnet50(opts,video)
                # print(len(videoFeature))

                # save feature to_csv
                audio_file = "audio/" + id + ".npy"
                # video_file = "video/"+id+".npy"
                audioFeaturePath = os.path.join(feature_path, audio_file)
                # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                np.save(audioFeaturePath, audioFeature)
                # np.save(videoFeaturePath,videoFeature)

        check_json[file] = 1
        save_json(check_json, json_path)


def cmumosei_data_embedding(opts):
    ids_path = opts.cmumosei_ids_path
    audio_path = opts.cmumosei_audio_path
    video_path = opts.cmumosei_video_path
    ids = np.array(pd.read_csv(ids_path))
    ids = ids.reshape(ids.shape[0], ).tolist()
    print("ids:", len(ids))
    if not os.path.exists(opts.csv_path):
        df = pd.DataFrame(columns=["audio_feature", "video_feature", "emotion_label"])
        df.to_csv(opts.csv_path, index=False)
    for id in tqdm(ids):
        print("id:", id)  # DNzA2UIkRZk_2 4142
        # load audio wava
        wave_data, samplerate = librosa.load(os.path.join(audio_path, id + ".wav"), sr=16000)
        print(wave_data.shape, "data length:", wave_data.shape[0] / samplerate)

        # read video images
        videodir = os.path.join(video_path, id + "_aligned")
        imglist = os.listdir(videodir)
        video = []
        for image in imglist:
            imgpath = os.path.join(videodir, image)
            img = Image.open(imgpath)
            video.append(np.array(img))
            img.close()

        # label
        label = random.randint(0, 7)

        audio_second = wave_data.shape[0] / samplerate  # audio length
        max_audio_second = opts.max_seq_length / opts.audiofeature_persecond  # max audio length
        segment_num = 1
        # audio length > max audio length
        if audio_second > max_audio_second:
            segment_num = math.ceil(audio_second / (max_audio_second - 1))
            for seg in range(segment_num):
                print("seg:", seg)
                # extract audio features
                a_start = int((seg * (max_audio_second - 1)) * samplerate)
                a_end = int(min(wave_data.shape[0], a_start + max_audio_second * samplerate))
                print("start:", a_start, "end:", a_end)
                # Discard less than the shortest speech duration
                if a_end - a_start < opts.min_audio_second * samplerate:
                    continue

                audioFeature = audio_Wav2Vec2(opts, wave_data[a_start:a_end])
                print("audioFeature.shape:", audioFeature.shape)

                # extract video feature
                video_length = len(video)
                v_start = math.floor(video_length * a_start / wave_data.shape[0])
                v_end = math.ceil(video_length * a_end / wave_data.shape[0])
                print("v_start:", v_start, "v_end:", v_end)
                videoFeature = video_resnet50(opts, video[v_start:v_end])
                # print("len(videoFeature):",len(videoFeature))

                # save feature to_csv
                audio_file = "audio/" + id + "_" + str(seg) + ".npy"
                video_file = "video/" + id + "_" + str(seg) + ".npy"
                audioFeaturePath = os.path.join(opts.feature_path, audio_file)
                videoFeaturePath = os.path.join(opts.feature_path, video_file)
                np.save(audioFeaturePath, audioFeature)
                np.save(videoFeaturePath, videoFeature)

                df = pd.DataFrame([[audio_file, video_file, label]])
                df.to_csv(opts.csv_path, mode="a", index=False, header=False)
                del df
        else:

            if wave_data.shape[0] < opts.min_audio_second * samplerate:
                continue

            audioFeature = audio_Wav2Vec2(opts, wave_data)
            # print(audioFeature.shape)

            videoFeature = video_resnet50(opts, video)
            # print(len(videoFeature))

            audio_file = "audio/" + id + ".npy"
            video_file = "video/" + id + ".npy"
            audioFeaturePath = os.path.join(opts.feature_path, audio_file)
            videoFeaturePath = os.path.join(opts.feature_path, video_file)
            np.save(audioFeaturePath, audioFeature)
            np.save(videoFeaturePath, videoFeature)

            df = pd.DataFrame([[audio_file, video_file, label]])
            df.to_csv(opts.csv_path, mode="a", index=False, header=False)
            del df
    print("cmumosei_data_embedding done")


if __name__ == "__main__":
    args = get_args()
    # cmumosei_data_embedding(args)
    # unlabeled_data_csv(args)
    unlabeled_audio_data_embedding(args)

# nohup python -u "/home/lishuai/workspace/UniAV/modules/feature_embedding.py" >feature_embedding.log 2>&1 &
# nohup python -u "/public/home/zwchen209/lishuai/UniAV/modules/feature_embedding.py" >output/unlabeled_audio_data_embedding.log 2>&1 &
# 25584
