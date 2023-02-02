import torch
# Path setting
DATASET_PATH = "dataset"
INPUTS_PATH = "original_midi"
OUTPUTS_PATH = "generate_midi"
# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl', '.midi', '.mid']
# 'chordloader.py'
TEMPERATURE = 0
RHYTHM_DENSITY = 0
LEAD_SHEET = True

# 'BLSTM parameters'
VAL_RATIO = 0.1
RNN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 256
EPOCHS = 10

# EG_VAE parameters
MT_train = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
readme_fn = './train.py'
batch_size = 128
n_epoch = 6
clip = 1
parallel = False
weights = [1, 0.5]
beta = 0.1
tf_rates = [(0.6, 0), (0.5, 0), (0.5, 0)]
lr = 1e-3
name = 'Duoluoluos'
parallel = parallel if torch.cuda.is_available() and \
                       torch.cuda.device_count() > 1 else False