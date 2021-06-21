import glob
import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_ts,
 get_speech_ts_adaptive,
 save_audio,
 read_audio,
 state_generator,
 single_audio_stream,
 collect_chunks) = utils

files_dir = torch.hub.get_dir() + ''

wav = read_audio(f'{files_dir}/en.wav')
# get speech timestamps from full audio file
speech_timestamps = get_speech_ts(wav, model,
                                  num_steps=4)
pprint(speech_timestamps)

# merge all speech chunks to one audio
save_audio('only_speech.wav',
           collect_chunks(speech_timestamps, wav), 16000)
Audio('only_speech.wav')

wav = read_audio(f'{files_dir}/en.wav')
# get speech timestamps from full audio file
speech_timestamps = get_speech_ts_adaptive(wav, model, step=500, num_samples_per_window=4000)
pprint(speech_timestamps)

# merge all speech chunks to one audio
save_audio('only_speech.wav',
           collect_chunks(speech_timestamps, wav), 16000)
Audio('only_speech.wav')

wav = f'{files_dir}/en.wav'

for batch in single_audio_stream(model, wav):
    if batch:
        pprint(batch)

audios_for_stream = glob.glob(f'{files_dir}/*.wav')
len(audios_for_stream) # total 4 audios

for batch in state_generator(model, audios_for_stream, audios_in_stream=2): # 2 audio stream
    if batch:
        pprint(batch)




