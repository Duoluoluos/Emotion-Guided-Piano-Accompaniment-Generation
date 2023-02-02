import os
import pickle
import numpy as np
from copy import deepcopy
from tqdm import trange
from music21 import *
from config import *

chord_index = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"] # only # instead of b
chord_Mm_check = [[[0,4,7],""],[[0,3,7],"m"]]
chord_tri_check = [[[0,4,7],""],[[0,3,7],"m"],[[0,4,8],"aug"],[[0,3,6],"dim"]]
chord_seven_check = \
    [[[0,4,7],""],[[0,3,7],"m"],[[0,4,8],"aug"],[[0,3,6],"dim"],
       [[0,4,7,10],"7"],[[0,4,7,11],"M7"],[[0,3,7,10],"m7"],[[0,3,7,11],"mM7",],
       [[0,3,6,9],"dim7"],[[0,3,6,10],"m7b5"],[[0,4,8,11],"aug7"]]     

class Chord_Loader:
    def __init__(self, recogLevel = "Mm"):
        self.recogLevel = recogLevel
        if recogLevel == "Mm":
            self.chord_check = chord_Mm_check[:]
        if recogLevel == "Tri":
            self.chord_check = chord_tri_check[:]
        if recogLevel == "Seven":
            self.chord_check = chord_seven_check[:]

    def chord_alu(self, x, scalar = 1):
        if x == len(self.chord_check) * len(chord_index):
            return x
        y = (x + scalar) % 12 + (x // 12) * 12
        return y
    def isChordEqual(self, x, y):
        return len(set(x) - set(y))
    def index2name(self, x):
        check_p = x // 12
        index_p = x % 12
        total = len(self.chord_check) * len(chord_index)
        if x == total:
            return "NC"
        if x > total or x < 0:
            return "NC"
        return chord_index[index_p] + self.chord_check[check_p][1]
    def name2index(self, name):
        if name is None:
            return len(self.chord_check) * len(chord_index)
        for i in range(len(self.chord_check)):
            for j in range(len(chord_index)):
                chord_name = chord_index[j] + self.chord_check[i][1]
                if name == chord_name:
                    return i * 12 + j
                if name == "NC":
                    return len(self.chord_check) * len(chord_index)
        return len(self.chord_check) * len(chord_index)
    def name2note(self, name, stage = 0):
        if name == "NC":
            return None
        obe_index = -1
        obe_check = -1 # the obeservation of the name
        for i in chord_index:
            for j in self.chord_check:
                std_chord_name = i + j[1]
                if std_chord_name == name:
                    obe_index = i
                    obe_check = j
                    break
        if obe_index == -1 and obe_check == -1:
            #print(name)
            return None
        else:
            re = obe_check[0][:]
            chord_num = chord_index.index(obe_index)
            for i in range(len(re)):
                re[i] = re[i] + chord_num + stage * 12
            return re
    def note2name(self, notes):
        re = None
        equalnum = 0.0
        for i in range(len(notes)):
            temp_notes = notes[:]
            for j in range(len(temp_notes)):
                temp_notes[j] = (temp_notes[j] - notes[i]) % 12
            for k in self.chord_check:
                temp_equal = len(k[0]) - self.isChordEqual(k[0],temp_notes)
                if temp_equal > equalnum:
                    re = chord_index[notes[i] % 12] + k[1]
                    equalnum = temp_equal
        return re





def ks2gap(ks):
    if isinstance(ks, key.KeySignature):
        ks = ks.asKey()

    try:
        # Identify the tonic
        if ks.mode == 'major':
            tonic = ks.tonic

        else:
            tonic = ks.parallel.tonic

    except:
        return interval.Interval(0)

    # Transpose score
    gap = interval.Interval(tonic, pitch.Pitch('C'))

    return gap


def get_filenames(input_dir):
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(input_dir):
        # Traverse the list of files
        for this_file in filelist:
            # Ensure that suffixes in the training set are valid
            if input_dir == DATASET_PATH and os.path.splitext(this_file)[-1] not in EXTENSION:
                continue
            filename = os.path.join(dirpath, this_file)
            filenames.append(filename)

    return filenames


def harmony2idx(element):
    # Extracts the MIDI pitch of each note in a harmony
    pitch_list = [sub_ele.pitch.midi for sub_ele in element.notes]
    pitch_list = sorted(pitch_list)

    bass_note = pitch_list[0] % 12
    quality = pitch_list[min(1, len(pitch_list) - 1)] - pitch_list[0]

    if quality <= 3:
        quality = 0

    else:
        quality = 1

    return bass_note * 2 + quality


def melody_reader(score):
    melody_vecs = []
    chord_list = []
    gap_list = []
    last_chord = 0
    last_ks = key.KeySignature(0)

    for m in score.recurse():
        if isinstance(m, stream.Measure):
            vec = [0] * 12
            if m.keySignature != None:
                gap = ks2gap(m.keySignature)
                last_ks = m.keySignature

            else:
                gap = ks2gap(last_ks)

            gap_list.append(gap)
            this_chord = None

            for n in m:
                if isinstance(n, note.Note):
                    # midi pitch as note onset
                    token = n.transpose(gap).pitch.midi

                elif isinstance(n, chord.Chord) and not isinstance(n, harmony.ChordSymbol):
                    notes = [n.transpose(gap).pitch.midi for n in n.notes]
                    notes.sort()
                    token = notes[-1]

                elif isinstance(n, harmony.ChordSymbol) and np.sum(vec) == 0:
                    this_chord = harmony2idx(n.transpose(gap)) + 1
                    last_chord = this_chord
                    continue

                else:
                    continue

                vec[token % 12] += float(n.quarterLength)

            if np.sum(vec) != 0:
                vec = np.array(vec) / np.sum(vec)
            melody_vecs.append(vec)

            if this_chord == None:
                this_chord = last_chord

            chord_list.append(this_chord)

    return melody_vecs, chord_list, gap_list


def convert_files(filename, fromDataset=True):
    print('\nConverting file {}'.format(filename))
    data_corpus = []
    score = converter.parse(filename)
    score = score.parts[0]
    if not fromDataset:
        original_score = deepcopy(score)
    song_data = []
    melody_vecs, chord_txt, gap_list = melody_reader(score)
    if fromDataset:
        song_data.append((melody_vecs, chord_txt))
    else:
        data_corpus.append((melody_vecs, gap_list, original_score, filename))
    if len(song_data) > 0:
        data_corpus.append(song_data)
    return data_corpus

                


