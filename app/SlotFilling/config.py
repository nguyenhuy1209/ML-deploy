import torch

INTENT_LABEL = ['UNK', 'report']
SLOT_LABEL = ['PAD', 'UNK', 'O', 'B-location', 'I-location', 'B-velocity', 'I-velocity', 'B-reason', 'I-reason']

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SL_MODEL_PATH = './model/JointIDSF_PhoBERTencoder/4e-5/0.15/100'
