from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoFeatureExtractor, ResNetModel, ResNetForImageClassification
import torch
from transformers import logging

#warning level 
logging.set_verbosity_warning()


class AudioWav2Vec2(nn.Module):
    def __init__(self,modelpath):
        super().__init__()
        self.tokenizer = Wav2Vec2Processor.from_pretrained(modelpath,padding=True)
        self.model = Wav2Vec2Model.from_pretrained(modelpath)

    def forward(self, wavdata):
        data = self.tokenizer(wavdata, return_tensors="pt",sampling_rate=16000, padding="longest").input_values
        feature = self.model(data).last_hidden_state
        #[1, 665, 768]
        feature = torch.squeeze(feature)# [665,768]
        return feature

class ResNet50(nn.Module):
    def __init__(self, modelpath):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(modelpath)
        self.model = ResNetModel.from_pretrained(modelpath)

    def forward(self, image):
        inputs = self.feature_extractor(image, return_tensors="pt")
        feature = self.model(**inputs).pooler_output
        #[1,2048,1,1]->[2048]
        feature = torch.squeeze(feature)
        return feature


def audio_Wav2Vec2(opts, wavdata):
    modelpath = opts.wav2vec2_base_960h
    wav2vec_model = AudioWav2Vec2(modelpath)
    feature = wav2vec_model(wavdata).detach().numpy()
    return feature
def video_resnet50(opts, images):
    modelpath = opts.resnet50
    resnet50_model = ResNet50(modelpath)
    features = []
    for image in images:
        feature = resnet50_model(image).detach().numpy()
        features.append(feature)
    return features