from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoFeatureExtractor, ResNetModel, ResNetForImageClassification
import torch
from transformers import logging
from torch.nn.init import xavier_uniform_, constant_

#warning level 
logging.set_verbosity_warning()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioWav2Vec2(nn.Module):
    def __init__(self,modelpath, device):
        super().__init__()
        self.device = device
        self.tokenizer = Wav2Vec2Processor.from_pretrained(modelpath,padding=True)
        self.model = Wav2Vec2Model.from_pretrained(modelpath).to(self.device)
        

    def forward(self, wavdata):
        data = self.tokenizer(wavdata, return_tensors="pt",sampling_rate=16000, padding="longest").input_values
        feature = self.model(data.to(self.device)).last_hidden_state
        #[1, 665, 768]
        feature = torch.squeeze(feature)# [665,768]
        return feature

# class ResNet50_old(nn.Module):
#     def __init__(self, modelpath, device):
#         super().__init__()
#         self.device = device
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained(modelpath)
#         self.model = ResNetModel.from_pretrained(modelpath).to(self.device)
#
#     def forward(self, image):
#         inputs = self.feature_extractor(image, return_tensors="pt")
#         feature = self.model(**inputs.to(self.device)).pooler_output
#         #[1,2048,1,1]->[2048]
#         feature = torch.squeeze(feature)
#         return feature

def get_layers(num_layers):
    if num_layers == 18:
        return [2, 2, 2, 2]
    elif num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]


class BottleNeck_IR(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       #nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
filter_list = [64, 64, 128, 256, 512]
class CBAMResNet_IR(nn.Module):
    def __init__(self, num_layers, feature_dim=512, drop_ratio=0.4, mode='ir',filter_list=filter_list):
        super(CBAMResNet_IR, self).__init__()
        assert num_layers in [18, 50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'se_ir', 'cbam_ir'], 'mode should be ir, se_ir or cbam_ir'
        layers = get_layers(num_layers)
        if mode == 'ir':
            block = BottleNeck_IR

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=1)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x

class Resnet50(nn.Module):

    def __init__(self, opts):
        super(Resnet50, self).__init__()

        self.num_classes = opts.num_classes
        self.feature_dim = opts.feature_dim
        self.drop_ratio = opts.drop_ratio

        self.backbone_block_expansion = 1
        self.backbone = CBAMResNet_IR(50, 512, mode='ir')

        layer_index = {
            'body.0': 'layer1.0',
            'body.1': 'layer1.1',
            'body.2': 'layer1.2',
            'body.3': 'layer2.0',
            'body.4': 'layer2.1',
            'body.5': 'layer2.2',
            'body.6': 'layer2.3',
            'body.7': 'layer3.0',
            'body.8': 'layer3.1',
            'body.9': 'layer3.2',
            'body.10': 'layer3.3',
            'body.11': 'layer3.4',
            'body.12': 'layer3.5',
            'body.13': 'layer3.6',
            'body.14': 'layer3.7',
            'body.15': 'layer3.8',
            'body.16': 'layer3.9',
            'body.17': 'layer3.10',
            'body.18': 'layer3.11',
            'body.19': 'layer3.12',
            'body.20': 'layer3.13',
            'body.21': 'layer4.0',
            'body.22': 'layer4.1',
            'body.23': 'layer4.2',
        }

        if opts.pretrain_backbone:
            print('use pretrain_backbone from face recognition model...')
            if device.type == 'cuda':
                ckpt = torch.load(opts.pretrain_backbone)
            if device.type == 'cpu':
                ckpt = torch.load(opts.pretrain_backbone, map_location=lambda storage, loc: storage)
            # self.backbone.load_state_dict(ckpt['net_state_dict'])

            net_dict = self.backbone.state_dict()
            pretrained_dict = ckpt

            new_state_dict = {}
            num = 0
            for k, v in pretrained_dict.items():
                num = num + 1
                print('num: {}, {}'.format(num, k))
                for k_r, v_r in layer_index.items():
                    if k_r in k:
                        if k[k_r.__len__()] == '.':
                            k = k.replace(k_r, v_r)
                            break

                if (k in net_dict) and (v.size() == net_dict[k].size()):
                    new_state_dict[k] = v
                else:
                    print('error...')

            un_init_dict_keys = [k for k in net_dict.keys() if k not in new_state_dict]
            print("un_init_num: {}, un_init_dict_keys: {}".format(un_init_dict_keys.__len__(), un_init_dict_keys))
            print("\n------------------------------------")

            for k in un_init_dict_keys:
                new_state_dict[k] = torch.DoubleTensor(net_dict[k].size()).zero_()
                if 'weight' in k:
                    if 'bn' in k:
                        print("{} init as: 1".format(k))
                        constant_(new_state_dict[k], 1)
                    else:
                        print("{} init as: xavier".format(k))
                        xavier_uniform_(new_state_dict[k])
                elif 'bias' in k:
                    print("{} init as: 0".format(k))
                    constant_(new_state_dict[k], 0)

            print("------------------------------------")

            self.backbone.load_state_dict(new_state_dict)

        # self.backbone.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.backbone.output_layer = nn.Sequential(nn.BatchNorm2d(512 * self.backbone_block_expansion),
                                          nn.Dropout(self.drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * self.backbone_block_expansion * opts.spatial_size * opts.spatial_size,
                                                    self.feature_dim),
                                          nn.BatchNorm1d(self.feature_dim))

        self.output = nn.Linear(self.feature_dim, self.num_classes)

        for m in self.backbone.output_layer.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.output.weight = xavier_uniform_(self.output.weight)
        self.output.bias = constant_(self.output.bias, 0)


    def forward(self, x):
        x = self.backbone(x)

        # return x
        x = self.output(x)
        return x



def audio_Wav2Vec2(opts, wavdata, device):
    modelpath = opts.wav2vec2_base_960h
    wav2vec_model = AudioWav2Vec2(modelpath)
    wav2vec_model.to(device)
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