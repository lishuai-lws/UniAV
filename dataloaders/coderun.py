import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)




from modules.module_embedding import audio_Wav2Vec2, video_resnet50
print("helloworld")