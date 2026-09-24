import os.path

import pandas as pd
from towhee import pipe, ops
import torch
from configs import args
import torchaudio
import tempfile

import towhee.hub
_original_get_op = towhee.hub._CACHE_MANAGER.get_operator


def _mock_get_operator(operator, tag, install_reqs, latest):
    # 当 Towhee 尝试拉取 vggish 时，直接返回我们刚才建好的本地路径，切断网络请求
    if "vggish" in operator:
        return '.towhee/operators/audio-embedding/vggish'
    # 对于其他算子（如 ffmpeg），保持它原本的运行逻辑
    return _original_get_op(operator, tag, install_reqs, latest)

# 替换底层的获取算子方法
towhee.hub._CACHE_MANAGER.get_operator = _mock_get_operator

import soundfile as sf
import numpy as np

import soundfile as sf
import numpy as np

def preprocess_audio_to_mono(input_path, target_sr=16000, keep_original_format=True):
    data, sample_rate = sf.read(input_path, dtype='float32')  # [T] or [T, C]
    
    # 取第一声道
    if data.ndim > 1:
        data = data[:, 0]  # [T]
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)

    info = sf.info(input_path)
    original_encoding = info.subtype  # 'PCM_16', 'PCM_24', 'FLOAT' 等

    if keep_original_format and 'PCM' in original_encoding:
        # 写 int16 PCM
        data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        sf.write(temp_path, data_int16, sample_rate, subtype='PCM_16')
    else:
        sf.write(temp_path, data, sample_rate, subtype='FLOAT')

    return temp_path

audio_vggish_pipeline = (  # pipeline building
     pipe.input('path')
     .map('path', 'frame', ops.audio_decode.ffmpeg())
     .map('frame', 'vecs', ops.audio_embedding.vggish(weights_path='.towhee/operators/audio-embedding/vggish/vggish.pth'))
     .output('vecs')
)

data_dir = args.data_dir


# test_id = 'zxis5LLvULw_12000_22000'
# test_path = f'{data_dir}/media/{test_id}/audio.wav'
# temp_path = preprocess_audio_to_mono(test_path)
# print(f"original audio info: {torchaudio.info(test_path)}")
# print(f"mono audio info: :{torchaudio.info(temp_path)}")
# test_embed = torch.tensor(audio_vggish_pipeline(temp_path).get()[0])
# print(test_embed.shape)
# os.unlink(temp_path)
#
#
# test_id = 'null_c-45AfEdAU050_99000_109000'
# test_path = f'{data_dir}/media/{test_id}/audio.wav'
# temp_path = preprocess_audio_to_mono(test_path)
# print(f"original audio info: {torchaudio.info(test_path)}")
# print(f"mono audio info: :{torchaudio.info(temp_path)}")
# test_embed = torch.tensor(audio_vggish_pipeline(temp_path).get()[0])
# print(test_embed.shape)
# os.unlink(temp_path)



metapath = os.path.join(data_dir, 'metadata.csv')
metadata = pd.read_csv(metapath, header=0)
metadata = metadata[metadata['split'].isin(['train', 'val', 'test_s', 'test_u', 'test_n'])]
# metadata = metadata[metadata['split'].isin(['test_s'])]

vids = metadata['uid'].apply(lambda x: x.rsplit('_', 2)[0]).unique()

save_dir = os.path.join(data_dir, 'audio_embed_correct')
os.makedirs(save_dir, exist_ok=True)

for vid in vids:
    audio_path = f'{data_dir}/media_correct/{vid}/audio.wav'
    temp_path = preprocess_audio_to_mono(audio_path)
    audio_embed = torch.tensor(audio_vggish_pipeline(temp_path).get()[0])
    os.unlink(temp_path)
    # print(f"{vid}: {audio_embed.shape}")
    torch.save(audio_embed, f'{save_dir}/{vid}.pt')
    print(f'{vid} embedding saved {audio_embed.shape}')

