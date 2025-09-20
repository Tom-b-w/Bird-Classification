import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa

# 读取原始音频（假设是单声道）
input_path = "D:/bird-whisperer/birdclef-preprocess/birdclef-preprocess/birdclef2023-dataset/audio_files\original/0034/294443_1.wav"
y, sr = librosa.load(input_path, sr=None, mono=True)
# 方法1：简单循环（适合环境音）
loop_times = 10  # 延长到约6秒
extended = np.tile(y, loop_times)
sf.write("./static/sounds/mallard_duck.wav", extended, sr)