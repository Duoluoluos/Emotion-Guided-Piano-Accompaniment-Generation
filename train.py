import warnings
warnings.simplefilter('ignore', UserWarning)
from models.model import DisentangleVAE
from amc_dl.torch_plus import DataLoaders
from dataset import prepare_dataset
from models.ptvae import RnnEncoder, TextureEncoder, PtvaeDecoderwithAtt, \
    RnnDecoder
from dataset import SEED
from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, MinExponentialLR, \
    TeacherForcingScheduler, ConstantScheduler
from amc_dl.torch_plus.train_utils import kl_anealing
import torch
from torch import optim
from models.multiloss import MGDA
from config import *

class MusicDataLoaders(DataLoaders):
    @staticmethod
    def get_loaders(seed, bs_train, bs_val,
                    portion=8, shift_low=-6, shift_high=5, num_bar=2,
                    contain_chord=True, random_train=True, random_val=False):
        train, val = prepare_dataset(seed, bs_train, bs_val, portion, shift_low,
                                     shift_high, num_bar, contain_chord,
                                     random_train, random_val)
        return MusicDataLoaders(train, val, bs_train, bs_val)

    def batch_to_inputs(self, batch):
        _, _, pr_mat, x, c, dt_x = batch
        pr_mat = pr_mat.to(self.device).float()
        x = x.to(self.device).long()
        c = c.to(self.device).float()
        dt_x = dt_x.to(self.device).float()
        return x, c, pr_mat, dt_x

chd_encoder = RnnEncoder(36, 1024, 256)
rhy_encoder = TextureEncoder(256, 1024, 256)
chd_decoder = RnnDecoder(z_dim=256)
pt_decoder = PtvaeDecoderwithAtt(note_embedding=None,dec_dur_hid_size=64,z_size=512)
model = DisentangleVAE(name, device, chd_encoder,
                       rhy_encoder, pt_decoder, chd_decoder).to(device)

# data loaders
data_loaders = \
    MusicDataLoaders.get_loaders(SEED, bs_train=batch_size, bs_val=batch_size,
                                 portion=8, shift_low=-6, shift_high=5,
                                 num_bar=2,
                                 contain_chord=True)

log_path_mng = LogPathManager(readme_fn)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

writer_names = ['loss', 'recon_loss', 'pl', 'dl', 'kl_loss', 'kl_chd',
                'kl_rhy', 'chord_loss', 'root_loss', 'valence_loss', 'bass_loss']
tags = {'loss': None}
summary_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)
tfr1_scheduler = TeacherForcingScheduler(*tf_rates[0])
tfr2_scheduler = TeacherForcingScheduler(*tf_rates[1])
tfr3_scheduler = TeacherForcingScheduler(*tf_rates[2])
weights_scheduler = ConstantScheduler(weights)
beta_scheduler = TeacherForcingScheduler(beta, 0., f=kl_anealing)
params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                  tfr3=tfr3_scheduler,
                  beta=beta_scheduler, weights=weights_scheduler)
param_scheduler = ParameterScheduler(**params_dic)
train_step = 0
eval_step = 0
epoch = 1

def loss_dic(loss_items):
    ans = {}
    assert len(writer_names) == len(loss_items)
    for key,val in zip(writer_names,loss_items):
        ans[key] = val.item()
    return ans

def train(parallel_training=False):
    model.train()
    param_scheduler.train()
    global  train_step
    print("length of train_loader:",len(data_loaders.train_loader))
    for i, batch in enumerate(data_loaders.train_loader):
        batch = data_loaders.batch_to_inputs(batch)
        optimizer_scheduler.optimizer_zero_grad()
        params = param_scheduler.step()
        tfr1,tfr2,tfr3,beta,weights = params.values()
        x,c,pr,_ = batch
        outputs = model.loss(x,c,pr,tfr1,tfr2,tfr3,beta,weights)
        loss = outputs[0]
        if MT_train==True:
            loss_lt=[outputs[1],outputs[4],outputs[7]]
            weights_vec = mt_trainer.backward(loss_lt, 'loss')
            print("weights:",weights_vec)
            loss = weights_vec[0]*outputs[1] + weights_vec[1]*outputs[4] + weights_vec[2]*outputs[7]
            outputs[0] = loss
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),optimizer_scheduler.clip)
        optimizer_scheduler.step()
        batch_loss_dic = loss_dic(outputs)
        summary_writers.write_task('train',batch_loss_dic,train_step)
        train_step+=1
    return loss

def eval(parallel_training=False):
    model.eval()
    param_scheduler.eval()
    global eval_step
    for i, batch in enumerate(data_loaders.val_loader):
        batch = data_loaders.batch_to_inputs(batch)
        params = param_scheduler.step()
        tfr1,tfr2,tfr3,beta,weights = params.values()
        x,c,pr,_ = batch
        with torch.no_grad():
            outputs = model.loss(x,c,pr,tfr1,tfr2,tfr3,beta,weights)
        loss = outputs[0]
        batch_loss_dic = loss_dic(outputs)
        summary_writers.write_task('val',batch_loss_dic,eval_step)
        eval_step+=1
    return loss

if __name__ == '__main__':
    torch.save(model.state_dict(), log_path_mng.model_path + "//NULLModel.pt")
    if MT_train:
        tasks=["recon_loss","kl_loss","chord_loss"]
        mt_trainer=MGDA(task_name=tasks, encoder_class=model, rep_grad=False, device='cpu')
    for i in range(1,n_epoch+1):
        train_loss = train()
        val_loss = eval()
        print("Epoch:{} Train_loss:{:.3f} Valid_Loss:{:.3f}".format(epoch,train_loss,val_loss))
        torch.save(model.state_dict(),log_path_mng.model_path+"//EmotionFlow-VAE-epoch{}.pt".format(epoch))
        epoch +=1