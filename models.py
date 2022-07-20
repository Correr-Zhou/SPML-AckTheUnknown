import torch
import numpy as np
import copy
from torchvision.models import resnet50
from collections import OrderedDict


'''
model definitions
'''


class FCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(FCNet, self).__init__()
        self.fc = torch.nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class ImageClassifier(torch.nn.Module):

    def __init__(self, P, model_feature_extractor=None, model_linear_classifier=None):

        super(ImageClassifier, self).__init__()
        print('initializing image classifier')

        model_feature_extractor_in = copy.deepcopy(model_feature_extractor)
        model_linear_classifier_in = copy.deepcopy(model_linear_classifier)

        self.arch = P['arch']

        if self.arch == 'resnet50':
            # configure feature extractor:
            if model_feature_extractor_in is not None:
                print('feature extractor: specified by user')
                feature_extractor = model_feature_extractor_in
            else:
                if P['use_pretrained']:
                    print('feature extractor: imagenet pretrained')
                    feature_extractor = resnet50(pretrained=True)
                else:
                    print('feature extractor: randomly initialized')
                    feature_extractor = resnet50(pretrained=False)
                feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
            if P['freeze_feature_extractor']:
                print('feature extractor frozen')
                for param in feature_extractor.parameters():
                    param.requires_grad = False
            else:
                print('feature extractor trainable')
                for param in feature_extractor.parameters():
                    param.requires_grad = True
            feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self.feature_extractor = feature_extractor

            # configure final fully connected layer:
            if model_linear_classifier_in is not None:
                print('linear classifier layer: specified by user')
                linear_classifier = model_linear_classifier_in
            else:
                print('linear classifier layer: randomly initialized')
                linear_classifier = torch.nn.Linear(P['feat_dim'], P['num_classes'], bias=True)
            self.linear_classifier = linear_classifier

        elif self.arch == 'linear':
            print('training a linear classifier only')
            self.feature_extractor = None
            self.linear_classifier = FCNet(P['feat_dim'], P['num_classes'])

        else:
            raise ValueError('Architecture not implemented.')

    def forward(self, x):
        if self.arch == 'linear':
            # x is a batch of feature vectors
            logits = self.linear_classifier(x)
        else:
            # x is a batch of images
            feats = self.feature_extractor(x)
            logits = self.linear_classifier(torch.squeeze(feats))
        return logits


class MultilabelModel(torch.nn.Module):
    def __init__(self, P, feature_extractor, linear_classifier):
        super(MultilabelModel, self).__init__()
        self.f = ImageClassifier(P, feature_extractor, linear_classifier)

    def forward(self, batch):
        f_logits = self.f(batch['image'])
        return f_logits
