#  导入模块许需要统一加入的，还没找到更好的解决办法
import csv
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from modules.module_embedding import audio_Wav2Vec2, video_resnet50
import argparse
import librosa
import os
import pandas as pd
import numpy as np
from PIL import Image
import math
# import cv2
import random
from transformers import logging

#修改告警显示级别
logging.set_verbosity_error()
from tqdm import tqdm




def get_args(description='data embedding'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--wav2vec2_base_960h", default="/home/lishuai/pretrainedmodel/wav2vec2-base-960h", help="pretrained wav2vect2.0 path")
    parser.add_argument("--resnet50", default="/home/lishuai/pretrainedmodel/resnet-50", help="pretrained resnet50 path")
    parser.add_argument("--cmumosei_ids_path",default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/ids.csv")
    parser.add_argument("--cmumosei_audio_path",default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/audio/WAV_fromVideo")
    parser.add_argument("--cmumosei_video_path",default="/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/video/version_img_size_224_img_scale_1.3")
    parser.add_argument("--csv_path",default="/home/lishuai/workspace/data/cmumosei.csv")
    parser.add_argument("--feature_path", default="/home/lishuai/workspace/feature/cmumosei")
    parser.add_argument("--max_seq_length",default=512,help=" max sequence length for encoder")
    parser.add_argument("--audiofeature_persecond",default=50,help=" wav2vec2.0 per second feature numbers")
    parser.add_argument("--min_audio_second",default=1,help="最短的语音时长，单位秒")
    parser.add_argument("--unlabeled_data_path", default="/public/home/zwchen209/Face-Detect-Track-Extract-main/output")
    parser.add_argument("--unlabeled_data_csv_path", default="/public/home/zwchen209/Face-Detect-Track-Extract-main/output/data/unlabeled_data.csv")

    args = parser.parse_args()

    return args
def unlabeled_data_csv(opts):
    data_path = opts.unlabeled_data_path
    csv_path = opts.unlabeled_data_csv_path
    audios_path = os.path.join(data_path, "audio")
    videos_path = os.path.join(data_path, "images")
    files_list = os.listdir(audios_path)
    df = pd.DataFrame(columns=["audio_feature","video_feature"])
    df.to_csv(csv_path,index=False)
    for file in tqdm(files_list):
        file_path = os.path.join(audios_path, file)
        audios_list = os.listdir(file_path)
        for audio in tqdm(audios_list):
            id  = audio[:-4]
            audio_path = os.path.join(file_path, audio)
            #读取音频数据
            wave_data, samplerate = librosa.load(audio_path, sr=16000)
            audio_second = wave_data.shape[0]/samplerate#语音时长
            max_audio_second = opts.max_seq_length/opts.audiofeature_persecond#最长的语音时长
                  #大于语音最大时长需要切分
            if audio_second>max_audio_second:
                segment_num = math.ceil(audio_second/(max_audio_second-1))
                for seg in range(segment_num):
                    print ("seg:",seg)
                    # 提取音频特征
                    a_start = int((seg*(max_audio_second-1))*samplerate)
                    a_end = int(min(wave_data.shape[0],a_start+max_audio_second*samplerate))
                    print("start:",a_start,"end:",a_end) 
                    #小于最短语音时长的舍弃掉
                    if a_end-a_start<opts.min_audio_second*samplerate:
                        continue

                    # audioFeature = audio_Wav2Vec2(opts,wave_data[a_start:a_end])
                    # print("audioFeature.shape:",audioFeature.shape)

                    # 提取视频特征
                    # video_length = len(video)
                    # v_start = math.floor(video_length*a_start/wave_data.shape[0])
                    # v_end = math.ceil(video_length*a_end/wave_data.shape[0])
                    # print("v_start:",v_start,"v_end:",v_end)
                    # videoFeature = video_resnet50(opts,video[v_start:v_end])
                    # print("len(videoFeature):",len(videoFeature))

                    #保存特征到文件
                    audio_file = "audio/"+id+"_"+str(seg)+".npy"
                    video_file = "video/"+id+"_"+str(seg)+".npy"
                    # audioFeaturePath = os.path.join(opts.feature_path,audio_file)
                    # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                    # np.save(audioFeaturePath,audioFeature)
                    # np.save(videoFeaturePath,videoFeature)

                    df = pd.DataFrame([[audio_file,video_file]])
                    df.to_csv(csv_path,mode="a",index=False, header=False)
                    del df #释放资源，7%,37%
            else :
                    #小于最短语音时长的舍弃掉
                    if wave_data.shape[0]<opts.min_audio_second*samplerate:
                        continue
                    # 提取音频特征
                    # audioFeature = audio_Wav2Vec2(opts,wave_data)
                    # print(audioFeature.shape)

                    # 提取视频特征
                    # videoFeature = video_resnet50(opts,video)
                    # print(len(videoFeature))

                    #保存特征到文件
                    audio_file = "audio/"+id+".npy"
                    video_file = "video/"+id+".npy"
                    # audioFeaturePath = os.path.join(opts.feature_path,audio_file)
                    # videoFeaturePath = os.path.join(opts.feature_path,video_file)
                    # np.save(audioFeaturePath,audioFeature)
                    # np.save(videoFeaturePath,videoFeature)

                    # df = pd.DataFrame([[audio_file,video_file,label]])
                    df = pd.DataFrame([[audio_file,video_file]])
                    df.to_csv(csv_path,mode="a",index=False, header=False)
                    del df #释放资源，7%,37%


def cmumosei_data_embedding(opts):
    ids_path = opts.cmumosei_ids_path
    audio_path = opts.cmumosei_audio_path
    video_path = opts.cmumosei_video_path
    ids = np.array(pd.read_csv(ids_path))
    ids = ids.reshape(ids.shape[0], ).tolist()
    print("ids:",len(ids))
    if not os.path.exists(opts.csv_path):
        df = pd.DataFrame(columns=["audio_feature","video_feature","emotion_label"])
        df.to_csv(opts.csv_path,index=False)
    for id in tqdm(ids):
        print("id:",id)#DNzA2UIkRZk_2 4142
        #读取音频数据
        wave_data, samplerate = librosa.load(os.path.join(audio_path, id + ".wav"), sr=16000)
        print(wave_data.shape,"data length:",wave_data.shape[0]/samplerate)

        #读取视频的图片数据
        videodir = os.path.join(video_path, id + "_aligned")
        imglist = os.listdir(videodir)
        video = []
        for image in imglist:
            imgpath = os.path.join(videodir, image)
            img = Image.open(imgpath)
            video.append(np.array(img))
            img.close()
        
        #获取label
        label = random.randint(0,7)#使用随机数代替标签

        audio_second = wave_data.shape[0]/samplerate#语音时长
        max_audio_second = opts.max_seq_length/opts.audiofeature_persecond#最长的语音时长
        segment_num = 1
        #大于语音最大时长需要切分
        if audio_second>max_audio_second:
            segment_num = math.ceil(audio_second/(max_audio_second-1))
            for seg in range(segment_num):
                print ("seg:",seg)
                # 提取音频特征
                a_start = int((seg*(max_audio_second-1))*samplerate)
                a_end = int(min(wave_data.shape[0],a_start+max_audio_second*samplerate))
                print("start:",a_start,"end:",a_end) 
                #小于最短语音时长的舍弃掉
                if a_end-a_start<opts.min_audio_second*samplerate:
                    continue

                audioFeature = audio_Wav2Vec2(opts,wave_data[a_start:a_end])
                print("audioFeature.shape:",audioFeature.shape)

                # 提取视频特征
                video_length = len(video)
                v_start = math.floor(video_length*a_start/wave_data.shape[0])
                v_end = math.ceil(video_length*a_end/wave_data.shape[0])
                print("v_start:",v_start,"v_end:",v_end)
                videoFeature = video_resnet50(opts,video[v_start:v_end])
                # print("len(videoFeature):",len(videoFeature))

                #保存特征到文件
                audio_file = "audio/"+id+"_"+str(seg)+".npy"
                video_file = "video/"+id+"_"+str(seg)+".npy"
                audioFeaturePath = os.path.join(opts.feature_path,audio_file)
                videoFeaturePath = os.path.join(opts.feature_path,video_file)
                np.save(audioFeaturePath,audioFeature)
                np.save(videoFeaturePath,videoFeature)

                df = pd.DataFrame([[audio_file,video_file,label]])
                df.to_csv(opts.csv_path,mode="a",index=False, header=False)
                del df #释放资源，7%,37%
        else :
                #小于最短语音时长的舍弃掉
                if wave_data.shape[0]<opts.min_audio_second*samplerate:
                    continue
                # 提取音频特征
                audioFeature = audio_Wav2Vec2(opts,wave_data)
                # print(audioFeature.shape)

                # 提取视频特征
                videoFeature = video_resnet50(opts,video)
                # print(len(videoFeature))

                #保存特征到文件
                audio_file = "audio/"+id+".npy"
                video_file = "video/"+id+".npy"
                audioFeaturePath = os.path.join(opts.feature_path,audio_file)
                videoFeaturePath = os.path.join(opts.feature_path,video_file)
                np.save(audioFeaturePath,audioFeature)
                np.save(videoFeaturePath,videoFeature)

                df = pd.DataFrame([[audio_file,video_file,label]])
                df.to_csv(opts.csv_path,mode="a",index=False, header=False)
                del df #释放资源，7%,37%
    print("cmumosei_data_embedding done")
if __name__=="__main__":
    args = get_args()
    cmumosei_data_embedding(args)


# 运行命令
# nohup python -u "/home/lishuai/workspace/UniAV/modules/feature_embedding.py" >feature_embedding.log 2>&1 &
# 25584

