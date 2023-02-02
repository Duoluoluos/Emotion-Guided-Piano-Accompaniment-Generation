import pretty_midi as pyd
import numpy as np 
import pandas as pd
from utils.acc_utils import split_phrases, chord_shift
import utils.format_converter_update as cvt
from utils.AccoMontage import find_by_length, dp_search, render_acc, ref_spotlight, get_texture_filter
import os
from music21 import *
from utils.chordloader  import  convert_files
from models.model import BiLSTM
from config import *
from tqdm import trange


class EG_ACC:
    def __init__(self, input_melody_folder, save_folder):
        self.input_melody_folder = input_melody_folder
        self.save_folder = save_folder
        self.chord_dictionary  = ['R','Cm', 'C','C#m', 'C#',
                    'Dm', 'D','D#m', 'D#','Em', 'E',
                    'Fm', 'F','F#m', 'F#','Gm', 'G',
                    'G#m', 'G#','Am', 'A','A#m', 'A#','Bm', 'B']

    def predict_chd(self,song, model):
        chord_list = []
        # Traverse the melody sequence
        for idx in range(int(len(song) / 4)):
            # Create input data
            melody = [song[idx * 4], song[idx * 4 + 1], song[idx * 4 + 2], song[idx * 4 + 3]]
            melody = np.array([np.array(seg) for seg in melody])
            x = torch.from_numpy(np.array(melody)).float()
            x = x.unsqueeze(0)
            y = model(x).squeeze(0)
            pred = y.max(axis=1, keepdim=True)[1].detach().numpy()
            for chord_idx in pred:
                chord_list.append(self.chord_dictionary[chord_idx[0]])

        # Create input data
        melody = [song[-4], song[-3], song[-2], song[-1]]
        melody = np.array([np.array(seg) for seg in melody])

        # Predict the last four chords
        x = torch.from_numpy(np.array(melody)).float()
        x = x.unsqueeze(0)
        y = model(x).squeeze(0)
        pred = y.max(axis=1, keepdim=True)[1].detach().numpy()
        for idx in range(-1 * (len(song) % 4), 0):
            chord_list.append(self.chord_dictionary[pred[idx][0]])
        return chord_list

    def export_music(self, score, chord_list, gap_list, filename):
        harmony_list = []
        filename = os.path.basename(filename)
        filename = '.'.join(filename.split('.')[:-1])

        for idx in range(len(chord_list)):
            chord = chord_list[idx]
            if chord == 'R':
                harmony_list.append(note.Rest())
            else:
                harmony_list.append(harmony.ChordSymbol(chord).transpose(-1 * gap_list[idx].semitones))
        m_idx = 0
        new_score = []
        for i in range(len(harmony_list)):
            new_m = stream.Measure(harmony_list[i])
            new_m.offset = i * 4.0
            new_score.append(new_m)
            # m_idx += 1
        new_score = stream.Stream(new_score)
        new_score[-1].rightBarline = bar.Barline('final')
        res = stream.Score()
        chd = stream.Part()
        chd.append(stream.Measure(new_score))
        chd.partName = 'Chord'
        res.insert(0, score)
        res.insert(0, chd)
        res.write('mid', fp=self.save_folder + '/' + filename + '_chd.mid')

    def Chorderator(self,chord_model,input_fs):
        print("Harmonizing the input melody ({})".format(input_fs))
        # Build model
        model = BiLSTM()
        check_point = torch.load(chord_model, map_location='cpu')
        model.load_state_dict(check_point)
        data_corpus = convert_files(input_fs, fromDataset=False)
        # Process each melody sequence
        for idx in trange(len(data_corpus)):
            melody_vecs = data_corpus[idx][0]
            gap_list = data_corpus[idx][1]
            score = data_corpus[idx][2]
            filename = data_corpus[idx][3]
            chord_list = self.predict_chd(melody_vecs, model)
            self.export_music(score, chord_list, gap_list, filename)
        print("Melody has been harmonized with chords")

    def texture_design(self,input_fs,SEGMENTATION,SPOTLIGHT=[],PREFILTER=(0,0),NOTE_SHIFT=0):
        print('Loading Reference Data')
        data = np.load('data/phrase_data0714.npz', allow_pickle=True)
        melody = data['melody']
        acc = data['acc']
        chord = data['chord']
        edge_weights=np.load('data/edge_weights_0714.npz', allow_pickle=True)
        print('Processing Query Lead Sheet')
        midi = pyd.PrettyMIDI(input_fs)
        melody_track, chord_track = midi.instruments[0], midi.instruments[1]
        downbeats = midi.get_downbeats()
        melody_matrix = cvt.melody_data2matrix(melody_track, downbeats)# T*130, quantized at 16th note
        if not NOTE_SHIFT == 0:
            melody_matrix = np.concatenate((melody_matrix[int(NOTE_SHIFT*4):, :], melody_matrix[-int(NOTE_SHIFT*4):, :]), axis=0)
        chroma = cvt.chord_data2matrix(chord_track, downbeats, 'quater')  # T*36, quantized at 16th note (quater beat)
        if not NOTE_SHIFT == 0:
            chroma = np.concatenate((chroma[int(NOTE_SHIFT*4):, :], chroma[-int(NOTE_SHIFT*4):, :]), axis=0)
        chord_table = chroma[::4, :] #T'*36, quantized at 4th notes
        chroma = chroma[:, 12: -12] #T*12, quantized at 16th notes
        pianoRoll = np.concatenate((melody_matrix, chroma), axis=-1)    #T*142, quantized at 16th
        query_phrases = split_phrases(SEGMENTATION) #[('A', 8, 0), ('A', 8, 8), ('B', 8, 16), ('B', 8, 24)]
        query_seg = [item[0] + str(item[1]) for item in query_phrases]  #['A8', 'A8', 'B8', 'B8']
        melody_queries = []
        for item in query_phrases:
            start_bar = item[-1]
            length = item[-2]
            segment = pianoRoll[start_bar*16: (start_bar+length)*16]
            melody_queries.append(segment)  #melody queries: list of T16*142, segmented by phrases

        print('Processing Reference Phrases')
        acc_pool = {}
        (mel, acc_, chord_, song_reference) = find_by_length(melody, acc, chord, 8)
        acc_pool[8] = (mel, acc_, chord_, song_reference)

        (mel, acc_, chord_, song_reference) = find_by_length(melody, acc, chord, 4)
        acc_pool[4] = (mel, acc_, chord_, song_reference)

        (mel, acc_, chord_, song_reference) = find_by_length(melody, acc, chord, 6)
        acc_pool[6] = (mel, acc_, chord_, song_reference)

        texture_filter = get_texture_filter(acc_pool)

        print('Phrase Selection Begins:\n\t', len(query_phrases), 'phrases in query lead sheet;\n\t', 'Refer to', SPOTLIGHT, 'as much as possible;\n\t', 'Set note density filter:', PREFILTER, '.')
        phrase_indice, chord_shift = dp_search(
                                                melody_queries,
                                                query_seg,
                                                acc_pool,
                                                edge_weights,
                                                texture_filter,
                                                filter_id=PREFILTER,
                                                spotlights=ref_spotlight(SPOTLIGHT))
        path = phrase_indice[0]
        shift = chord_shift[0]
        reference_set = []
        df = pd.read_excel("data/POP909 4bin quntization/four_beat_song_index.xlsx")
        for idx_phrase, phrase in enumerate(query_phrases):
            phrase_len = phrase[1]
            song_ref = acc_pool[phrase_len][-1]
            idx_song = song_ref[path[idx_phrase][0]][0]
            song_name = df.iloc[idx_song][1]
            reference_set.append((idx_song, song_name))
        print('Generating...')
        midi = render_acc(pianoRoll, chord_table, query_seg, path, shift, acc_pool, 'data/Poly-dis-VAE-epoch4.pt')
        midi.write(self.save_folder + '/' + input_fs.split('/')[-1].split('.')[-2]+'_acc.mid')
        print('Result saved')

    def solve(self):
        input_fs = os.listdir(self.input_melody_folder)
        output_fs = os.listdir(self.save_folder)
        for music in input_fs:
            self.Chorderator('data/blstm.pt', self.input_melody_folder +'/'+music)
            self.texture_design(input_fs = self.save_folder +'/'+music.split('.')[-2] + '_chd.mid'
                                 ,SPOTLIGHT=[], SEGMENTATION='A8B8A8B8\n')