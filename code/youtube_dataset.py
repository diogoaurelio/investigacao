import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

YOUTUBE_DATASETS = '~/Documents/Development/phd/youtube_personality'
audio_visual_features_file = '{}/YouTube-Personality-audiovisual_features.csv'.format(YOUTUBE_DATASETS)
gender_file = '{}/YouTube-Personality-gender.csv'.format(YOUTUBE_DATASETS)
impression_scores_file = '{}/YouTube-Personality-Personality_impression_scores.csv'.format(YOUTUBE_DATASETS)

av = pd.read_csv(audio_visual_features_file)
gender = pd.read_csv(gender_file)
impression_scores = pd.read_csv(impression_scores_file)
