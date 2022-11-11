# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import logging
import io
import os
import sys
import json
import struct

import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, List, Optional, Union, IO
from image_synthesis.data.utils import data_utils
from image_synthesis.data.utils.dictionary import Dictionary

ENDIAN = 'little'

logger = logging.getLogger(__name__)

_encodec_header_struct = struct.Struct('!4sBI')
_ENCODEC_MAGIC = b'ECDC'



def _read_exactly(fo: IO[bytes], size: int) -> bytes:
    buf = b""
    while len(buf) < size:
        new_buf = fo.read(size)
        if not new_buf:
            raise EOFError("Impossible to read enough data from the stream, "
                           f"{size} bytes remaining.")
        buf += new_buf
        size -= len(new_buf)
    return buf

def read_ecdc_header(fo: IO[bytes]):
    header_bytes = _read_exactly(fo, _encodec_header_struct.size)
    magic, version, meta_size = _encodec_header_struct.unpack(header_bytes)
    if magic != _ENCODEC_MAGIC:
        raise ValueError("File is not in ECDC format.")
    if version != 0:
        raise ValueError("Version not supported.")
    meta_bytes = _read_exactly(fo, meta_size)
    return json.loads(meta_bytes.decode('utf-8'))

class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.
    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        """
    def __init__(self, bits: int, fo: IO[bytes]):
        self.bits = bits
        self.fo = fo
        self._mask = (1 << bits) - 1
        self._current_value = 0
        self._current_bits = 0

    def pull(self) -> Optional[int]:
        """
        Pull a single value from the stream, potentially reading some
        extra bytes from the underlying file-object.
        Returns `None` when reaching the end of the stream.
        """
        while self._current_bits < self.bits:
            buf = self.fo.read(1)
            if not buf:
                return None
            character = buf[0]
            self._current_value += character << self._current_bits
            self._current_bits += 8

        out = self._current_value & self._mask
        self._current_value >>= self.bits
        self._current_bits -= self.bits
        return out



class AudioLMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        prompt_data_root=None,
        phase="train",
        prev_sample_rate=20,
        sample_rate=20,
        tokens_per_sample=None,
        tokens_per_frame=8,
        condition_seq_len=200,
        stage=1,
    ):
        self._data_root = data_root
        self._prompt_data_root = prompt_data_root
        self._prev_sample_rate = prev_sample_rate
        self._sample_rate = sample_rate
        self._label_ratio = self._prev_sample_rate / self._sample_rate

        self.tokens_per_sample = tokens_per_sample
        self.tokens_per_frame = tokens_per_frame
        self.condition_seq_len = condition_seq_len
        self.stage = stage

        self.prompt_data = []
        self.prompt_size = []
        self.data = []
        self.sizes = []        
        
        with open(os.path.join(self._data_root, phase+'.tsv')) as f:
            self.root = f.readline().strip()
            for line in f:
                self.data.append(line.strip().split('\t')[0])
                size = int(line.strip().split('\t')[1])
                self.sizes.append(size // self.tokens_per_frame)

        self.dictionary = Dictionary.load(
            os.path.join(os.path.dirname(self._data_root), 'dict.txt')
        )
        if stage == 0 or stage == 1:
            self.prompt_dictionary = Dictionary.load(
                os.path.join(os.path.dirname(self._prompt_data_root), 'dict.txt'))
                
            self.prompt_dataset = data_utils.MMapIndexedDataset(
                os.path.join(self._prompt_data_root, phase)
            )
            assert len(self.prompt_dataset) == len(self.data)


    def __len__(self):
        return len(self.data)

    def num_tokens(self, index):
        return self.size(index) * self.tokens_per_frame // 2

    def size(self, index):
        if self.tokens_per_sample is None:
            return self.sizes[index]
        else:
            return min(self.sizes[index], self.tokens_per_sample)


    def __getitem__(self, index):
        chunk = self.data[index]
        parts = chunk.split(':')
        chunk_name = parts[0]
        start = int(parts[1])
        length = int(parts[2])
        data = []
        if self.stage == 0 or self.stage == 1:
            prompt = self.prompt_dataset[index]

        with open(os.path.join(self.root, chunk_name), 'rb') as f:
            f.seek(start)
            data = f.read(length)
        
        fo = io.BytesIO(data)
        metadata = read_ecdc_header(fo)
        audio_length = metadata['al']
        num_codebooks = metadata['nc']
        unpacker = BitUnpacker(10, fo)
        frame_length = int(audio_length / 24000 * 75)
        frame = torch.zeros(num_codebooks, int(frame_length), dtype=torch.long)
        for t in range(frame_length):
            code_list = []
            for k in range(num_codebooks):
                code = unpacker.pull() + self.dictionary.nspecial
                code_list.append(code)
            codes = torch.LongTensor(code_list)
            frame[:, t] = codes
        
        if self.stage == 1:
            tokens = frame[:self.tokens_per_frame // 2, :]
        elif self.stage == 2:
            tokens = frame[self.tokens_per_frame//2:, :]
        else:
            tokens = frame

        if tokens.shape[-1] > self.tokens_per_sample:
            diff = tokens.shape[-1] - self.tokens_per_sample
            start = np.random.randint(0, diff+1)
            end = start + self.tokens_per_sample
            tokens = tokens[:, start:end]
            if self.stage ==0 or self.stage == 1:
                prompt = prompt[int(start*self._label_ratio): int(end*self._label_ratio)]
        else:
            tokens = torch.nn.functional.pad(tokens, pad=(0, self.tokens_per_sample-tokens.shape[1]), value=self.dictionary.pad_index, )
            
        tokens = tokens.transpose(0, 1).reshape(-1)
    
        if self.stage == 0 or self.stage == 1:
            prompt = torch.unique_consecutive(prompt)
            if len(prompt) > self.condition_seq_len:
                prompt = prompt[:self.condition_seq_len]
            else:
                prompt = torch.nn.functional.pad(prompt, pad=(0, self.condition_seq_len-len(prompt)), value=self.prompt_dictionary.pad_index)
            batch = {
                    'prompt': prompt,
                    'src_tokens': tokens
            }
        else:
            batch = {
                'src_tokens': tokens
            }
        return batch





    
