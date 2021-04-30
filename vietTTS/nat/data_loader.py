import random
from pathlib import Path

import numpy as np
import textgrid
from vietTTS.nat.model import DurationModel

from .config import DurationInput


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


def textgrid_data_loader(data_dir: Path, seq_len:int, batch_size: int, mode: str):
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
    ps, ds = zip(*load_textgrid(fn))
    ps = [phonemes.index(p) for p in ps]
    l = len(ps)
    ps = pad_seq(ps, seq_len, 0)
    ds = pad_seq(ds, seq_len, 0)
    data.append((ps, ds, l))

  batch = []
  while True:
    random.shuffle(data)
    for e in data:
      batch.append(e)
      if len(batch) == batch_size:
        ps, ds, lengths = zip(*batch)
        ps = np.array(ps, dtype=np.int32)
        ds = np.array(ds, dtype=np.float32) * 10
        lengths = np.array(lengths, dtype=np.int32)
        yield DurationInput(ps, lengths, ds)
        batch = []