from pathlib import Path

from effortless_config import Config, setting
import numpy as np
import soundfile as sf
import torch
from udls import SimpleDataset, simple_audio_preprocess
from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop

from rave.core import random_phase_mangle

if __name__ == "__main__":

    class args(Config):
        PREPROCESSED = None
        WAV = None
        SR = 48000
        N_SIGNAL = 65536
    
    args.parse_args()

    preprocess = lambda name: simple_audio_preprocess(
        args.SR,
        2 * args.N_SIGNAL,
    )(name).astype(np.float16)

    dataset = SimpleDataset(
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=preprocess,
        split_set="full",
        transforms=Compose([
            lambda x: x.astype(np.float32),
            RandomCrop(args.N_SIGNAL),
            RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
                p=.8,
            ),
            Dequantize(16),
            lambda x: x.astype(np.float32),
        ]),
    )

    outdir = Path(args.PREPROCESSED)
    for i, x in enumerate(dataset):
        outfile = Path(f"audio_{i:04d}.wav")
        sf.write(str(outdir.joinpath(outfile)), x, args.SR)
