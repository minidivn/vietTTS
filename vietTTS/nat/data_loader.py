import random
from pathlib import Path

import numpy as np
import textgrid
from scipy.io import wavfile

from .config import AcousticInput, DurationInput


def load_phonemes_set_from_lexicon_file(fn: Path):
  S = set()
  for line in open(fn, 'r').readlines():
    word, phonemes = line.strip().lower().split('\t')
    phonemes = phonemes.split()
    S.update(phonemes)

  S = ['sil', 'sp', 'spn'] + sorted(list(S))
  return S


def pad_seq(s, maxlen, value=0):
  assert maxlen >= len(s)
  return tuple(s) + (value,) * (maxlen - len(s))


def load_textgrid(fn: Path):
  tg = textgrid.TextGrid.fromFile(str(fn.resolve()))
  data = []
  for p in tg[1]:
    data.append((p.mark.strip().lower(), p.duration()))
  return data


def load_textgrid_wav(data_dir: Path, token_seq_len: int, batch_size, pad_wav_len, mode: str):
  tg_files = sorted(data_dir.glob('*.TextGrid'))
  random.Random(42).shuffle(tg_files)
  L = len(tg_files) * 8 // 10
  assert mode in ['train', 'val']
  phonemes = load_phonemes_set_from_lexicon_file(data_dir / 'lexicon.txt')
  if mode == 'train':
    tg_files = tg_files[:L]
  if mode == 'val':
    tg_files = tg_files[L:]

  data = []
  for fn in tg_files:
    wav_file = data_dir / f'{fn.stem}.wav'
    sr, y = wavfile.read(wav_file)

    ps, ds = zip(*load_textgrid(fn))
    pad_left, pad_right = 0, 0
    ps_idx = [phonemes.index(p) for p in ps]
    if ps[0] in ['sil', 'sp', 'spn']:
      ps_idx.pop(0)
      pad_left = int(ds[0] * sr)
      ds.pop(0)
    if ps[-1] in ['sil', 'sp', 'spn']:
      ps_idx.pop()
      pad_right = int(ds[-1] * sr)
      ds.pop()
    y = y[pad_left:(len(y) - pad_right)]

    if len(y) > pad_wav_len:
      y = y[:pad_wav_len]
    wav_length = len(y)
    y = np.pad(y, (0, pad_wav_len - len(y)))

    l = len(ps_idx)
    ps_idx = pad_seq(ps_idx, token_seq_len, 0)
    ds = pad_seq(ds, token_seq_len, 0)

    data.append((ps_idx, ds, l, y, wav_length))

  batch = []
  while True:
    random.shuffle(data)
    for e in data:
      batch.append(e)
      if len(batch) == batch_size:
        ps, ds, lengths, wavs, wav_lengths = zip(*batch)
        ps = np.array(ps, dtype=np.int32)
        ds = np.array(ds, dtype=np.float32)
        lengths = np.array(lengths, dtype=np.int32)
        wavs = np.array(wavs)
        wav_lengths = np.array(wav_lengths, dtype=np.int32)
        yield AcousticInput(ps, lengths, ds, wavs, wav_lengths, None)
        batch = []
