import pypianoroll as pr
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
def plot_pr(fs):
    multitrack=pr.read(fs)
    multitrack.plot(xtick='off',grid_axis='off')

def basic_metrics(fs):
    file=pr.read(fs)
    sc_rate = pr.qualified_note_rate(file.tracks[1].pianoroll)
    poly_rate = pr.empty_beat_rate(file.tracks[1].pianoroll,1)
    print(" scale_rate:{:.4f}% \n emp_rate:{:.4%}".format(sc_rate,poly_rate))

def plot_fig(X,Y,save_name,title):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=200)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", lw=0.5, color="#4E616C")
    ax.plot(X, Y,color='red',linewidth=1.5)
    ax.set_title(title)
    if save_name != None:
        plt.savefig("{}.png".format(save_name),dpi=400)
    else:
        plt.show()

def plot_jointdis(gener,templ,img_style='flare'):
    data = pd.DataFrame()
    data['Generated Valence Curve'] = gener
    data['User input'] = templ
    sns.set_palette(img_style)
    g = sns.JointGrid(data=data,x='Generated Valence Curve', y='User input')
    g.plot_joint(sns.histplot)
    g.plot_marginals(sns.boxplot)
    plt.savefig("Low Valence Basis", dpi=400)
    plt.show()

def valence_evaluation(chroma):
    chromas_templ = {
        #           1     2     3     4  5     6     7
        'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        'NC':      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    valence_templ = {
        'maj':1,'min':-1,'aug':0.5,'dim':0.5,'7':2,'maj7':2.5,
        'min7':0.5,'minmaj7':1.5,'dim7':0.2,'hdim7':1,'NC':0,
    }
    valence = []
    dur=chroma.shape[0]
    for i in range(0,dur,32):  #全音符为一个小节
        splitted_acc = chroma[i:] if dur-i<32 else chroma[i:i+32]
        temp = [0 for _ in range(32)]
        for j in range(splitted_acc.shape[0]):
            chord=splitted_acc[j,:]
            skewing = np.where(chord==1)[0][0] if 1 in chord else 0
            chord = np.roll(chord,-skewing)
            chord=chord.tolist()
            try:
                chord_key = list(chromas_templ.keys())[list(chromas_templ.values()).index(chord)]
            except:
                chord_key = 'NC'
            temp[j] = valence_templ[chord_key]
        valence.append(sum(temp)/24)
    # 随机测试，绘制一张联合概率热度+箱型图
    random.seed(1)
    original = valence.copy()
    random.shuffle(valence[len()])
    #plot_fig(list(range(len(arousal))), arousal, None, 'test_arousal')
    #plot_jointdis(original,valence)

def arousal_evaluation(acc_lt):
    arousal = []
    for acc in acc_lt:
        dur = acc.shape[0]
        for i in range(0,dur,16):  #两个全音符为一个小节
            splitted_acc = acc[i:] if dur-i<16 else acc[i:i+16]
            density = np.sum(np.reshape(splitted_acc,(splitted_acc.size,)))/64
            arousal.append(density)
    #plot_fig(list(range(len(arousal))),arousal,None,'test_arousal')
    # 随机测试，绘制一张联合概率热度+箱型图
    random.seed(1)
    original = arousal.copy()
    random.shuffle(arousal)
    #plot_fig(list(range(len(arousal))), arousal, None, 'test_arousal')
    #plot_jointdis(original,arousal)

if __name__=='__main__':
    basic_metrics("generate_midi/hpp7/M-CTDA-VAE-hpp7.mid")
