# Emotion-Guided-Piano-Accompaniment-Generation
This is the codes repository for the project "Emotion-guided Piano Accompaniment Generation". 

Baseline models are from [Accomontage Repo](https://github.com/zhaojw1998/AccoMontage).

Training and validation set is derived from [POP909 dataset](https://github.com/music-x-lab/POP909-Dataset) and
emotion-guided music generation test set is a part of [Nottingham Dataset](https://ifdo.ca/~seymour/nottingham/nottingham.html).

Harmonization module is updated  (2023/2/2)
The paper is accepted at International Joint Conference on Neural Networks 2023(IJCNN2023).
## Paper Link
[https://arxiv.org/abs/2307.04015](https://arxiv.org/abs/2307.04015).


## Run Demos
 1. [Click here](https://drive.google.com/drive/folders/1nw35zbV6wLCBggJiMyeBIxzAqzNwz_AR?usp=drive_link) to download the data folder and put them in `./data`.
   
 2. Put the melodies (MIDI Format) in the `./original_midi`;

 3. Run `demo.py`;

 4. Wait a while and the accompaniment will be saved in the `./generate_midi`.

 5. For more emotional flow guided generation samples, please refer to [https://soundcloud.com/ko9isjyplxrb/sets/demos-of-emotion-guided-generated-accompaniment](https://soundcloud.com/ko9isjyplxrb/sets/demos-of-emotion-guided-generated-accompaniment).

## Train the model
 1. [Click here](https://drive.google.com/drive/folders/1P9hx0WltA_kdDqiswaiWd-KspDROP_QK?usp=sharing) to download the processed dataset and put them in `./data`.
    
 2. Run `train.py`
    
 3. Log files will be automatically generated to the root directory, please use tensorboard related commands to monitor the training process.
    
## Q&A
1. Q: Sometimes the generated chords dont fully harmonize the melody. How come?
   
   A: The randomness and controllabilty of LSTM-based harmonizer isnt satisfactory. **Use music theory** for the harmonization instead. You can rectify the `solve()` function from
`./utils/Accompaniment_Generator.py` for your personal usage.
