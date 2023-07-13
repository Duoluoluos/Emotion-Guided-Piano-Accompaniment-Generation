# Emotion-Guided-Piano-Accompaniment-Generation
This is the codes repository for the project "Emotion-guided Piano Accompaniment Generation". 

Baseline models are from [Accomontage Repo](https://github.com/zhaojw1998/AccoMontage).

Training and validation set is derived from [POP909 dataset](https://github.com/music-x-lab/POP909-Dataset) and
emotion-guided music generation test set is a part of [Nottingham Dataset](https://ifdo.ca/~seymour/nottingham/nottingham.html).

Harmonization module is updated  (2023/2/2)
The paper is accepted at International Joint Conference on Neural Networks 2023(IJCNN2023), [here](https://arxiv.org/abs/2307.04015) is the paper link.
First, [click here](https://drive.google.com/file/d/1TAQymlAQQQpHfny-uqkYACrJf-fMS5ln/view?usp=sharing) to download the data folder. 
UI module will be completed in the future.

## Run Demos
1.　Put the melodies (MIDI Format) in the `original` folder;\
2.　Run `demo.py`;\
3.　Wait a while and the accompaniment will be saved in the `generate_midi` folder.\
4.　For more emotional flow guided generation samples, please refer to [this website](https://soundcloud.com/ko9isjyplxrb/sets/demos-of-emotion-guided-generated-accompaniment).
