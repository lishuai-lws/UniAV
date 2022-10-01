#  导入模块许需要统一加入的，还没找到更好的解决办法
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

    args = parser.parse_args()

    return args


def cmumosei_data_embedding(opts):
    ids_path = opts.cmumosei_ids_path
    audio_path = opts.cmumosei_audio_path
    video_path = opts.cmumosei_video_path
    ids = np.array(pd.read_csv(ids_path))
    ids = ids.reshape(ids.shape[0], ).tolist()
    df = pd.DataFrame(columns=["audio_feature","video_feature","emotion_label"])
    df.to_csv(opts.csv_path,index=False)
    for id in tqdm(ids[4143:4144]):
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
                audioFeature = audio_Wav2Vec2(opts,wave_data[a_start:a_end])
                print("audioFeature.shape:",audioFeature.shape)

                # 提取视频特征
                video_length = len(video)
                v_start = math.floor(video_length*a_start/wave_data.shape[0])
                v_end = math.ceil(video_length*a_end/wave_data.shape[0])
                print("v_start:",v_start,"v_end:",v_end)
                videoFeature = video_resnet50(opts,video[v_start:v_end])
                print("len(videoFeature):",len(videoFeature))

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
            # 提取音频特征
                audioFeature = audio_Wav2Vec2(opts,wave_data)
                print(audioFeature.shape)

                # 提取视频特征
                videoFeature = video_resnet50(opts,video)
                print(len(videoFeature))

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
    # cmumosei_data_embedding(args)
    file_path = os.path.join(args.feature_path,"video")
    file_list = os.listdir(file_path)
    data = np.load(os.path.join(file_path,file_list[0]))
    print("data:",data.shape)


# 运行命令
# nohup python -u "/home/lishuai/UniVL-AV/modules/feature_embedding.py" >feature_embedding.log 2>&1 &
# 25584

