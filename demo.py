from utils.Accompaniment_Generator import EG_ACC
import os
from config import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#arousal:0~100
My_acc = EG_ACC(input_melody_folder=INPUTS_PATH,save_folder=OUTPUTS_PATH)
My_acc.solve(arousal=10)
