import pretty_midi as pyd
import numpy as np 
import pandas as pd
import datetime
from utils.evaluation import valence_evaluation
from utils.acc_utils import split_phrases, chord_shift
import utils.format_converter_update as cvt
from utils.AccoMontage import find_by_length, dp_search, render_acc, ref_spotlight, get_texture_filter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def ACC_Infer(SPOTLIGHT=[],PREFILTER=(0,0),
              input_melody_fs="D://research//M-CTDA-VAE//data//original_midi//ECNU University Song.mid",
              SEGMENTATION='A8A8A8A8\n',NOTE_SHIFT=0):
    SONG_ROOT='./demo/demo lead sheets'
    print('Loading Reference Data')
    data = np.load('./data/phrase_data0714.npz', allow_pickle=True)
    melody = data['melody']
    acc = data['acc']
    chord = data['chord']
    edge_weights=np.load('./data/edge_weights_0714.npz', allow_pickle=True)
    print('Processing Query Lead Sheet')
    midi = pyd.PrettyMIDI(input_melody_fs)
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
    valence_evaluation(chroma)

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
    df = pd.read_excel("./data/POP909 4bin quntization/four_beat_song_index.xlsx")
    for idx_phrase, phrase in enumerate(query_phrases):
        phrase_len = phrase[1]
        song_ref = acc_pool[phrase_len][-1]
        idx_song = song_ref[path[idx_phrase][0]][0]
        song_name = df.iloc[idx_song][1]
        reference_set.append((idx_song, song_name))
    print('Reference chosen:', reference_set)
    print('Pitch Transpositon (Fit by Model):', shift)
    
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = 'generate_midi/' + time + '.mid'
    print('Generating...')
    midi = render_acc(pianoRoll, chord_table, query_seg, path, shift, acc_pool,'data/model_master_final.pt')
    #midi.write(save_path)
    print('Result saved at', save_path)
