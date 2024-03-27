import torch

torch.set_grad_enabled(False)

from einops import rearrange
from tqdm import tqdm

from rave import RAVE
from rave.core import search_for_run

from effortless_config import Config
from os import path, makedirs, environ
from pathlib import Path

import librosa as li

import GPUtil as gpu

import soundfile as sf
import cached_conv as cc


class args(Config):
    CKPT = None  # PATH TO YOUR PRETRAINED CHECKPOINT
    WAV_FOLDER = None  # PATH TO YOUR WAV FOLDER
    OUT = "./reconstruction/"
    BLOCK_SIZE = None
    HOP_SIZE = None
    WINDOW = False

cc.use_cached_conv(True)

args.parse_args()

# GPU DISCOVERY
CUDA = gpu.getAvailable(maxMemory=.05)
if len(CUDA):
    environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
    use_gpu = 1
elif torch.cuda.is_available():
    print("Cuda is available but no fully free GPU found.")
    print("Reconstruction may be slower due to concurrent processes.")
    use_gpu = 1
else:
    print("No GPU found.")
    use_gpu = 0

device = torch.device("cuda:0" if use_gpu else "cpu")

# LOAD RAVE
rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

# COMPUTE LATENT COMPRESSION RATIO
x = torch.randn(1, 1, 2**14).to(device)
z = rave.encode(x)
ratio = x.shape[-1] // z.shape[-1]

# SEARCH FOR WAV FILES
audios = tqdm(list(Path(args.WAV_FOLDER).rglob("*.wav")))

# RECONSTRUCTION
makedirs(args.OUT, exist_ok=True)
for audio in audios:

    # Reset cache
    x = torch.zeros(1, 1, 2**14).to(device)
    _ = rave.decode(rave.encode(x))

    audio_name = path.splitext(path.basename(audio))[0]
    audios.set_description(audio_name)

    # LOAD AUDIO TO TENSOR
    x, sr = li.load(audio, sr=rave.sr)
    x = torch.from_numpy(x).reshape(1, 1, -1).float().to(device)

    # PAD AUDIO
    n_sample = x.shape[-1]
    pad = (ratio - (n_sample % ratio)) % ratio
    x = torch.nn.functional.pad(x, (0, pad))

    # Block-based processing
    block_size = int(args.BLOCK_SIZE) if args.BLOCK_SIZE is not None else x.shape[-1]
    hop_size = int(args.HOP_SIZE) if args.HOP_SIZE is not None else block_size

    # Pad the start of the audio if the block size is larger than the hop size
    # This simulates a causal processing of the audio
    if block_size > hop_size:
        pad = block_size - hop_size
        x = torch.nn.functional.pad(x, (pad, 0))
    elif hop_size > block_size:
        assert hop_size % block_size == 0, "Hop size must be a multiple of the block size."

    x = x.unsqueeze(1)
    blocks = torch.nn.functional.unfold(x, (1, block_size), stride=(1, hop_size))
    blocks = rearrange(blocks, 'b w t -> t b 1 w')

    y = []
    for block in blocks:
        out = rave.decode(rave.encode(block))
        y.append(out)
    
    # need to overlap and add
    if hop_size < ratio:
        y = torch.cat(y, dim=1)
        y = rearrange(y, 'b t f -> b f t')
        if args.WINDOW:
            y = y * torch.hann_window(y.shape[1], device=y.device)[None, :, None]

        out_size = hop_size * (blocks.shape[0] - 1) + block_size
        y = torch.nn.functional.fold(y, (1, out_size), kernel_size=(1, 2048), stride=(1, hop_size))
        y = y.squeeze(2)
    else:
        y = torch.cat(y, dim=-1)

    y = y.reshape(-1).cpu().numpy()[:n_sample]

    # WRITE AUDIO
    sf.write(path.join(args.OUT, f"{audio_name}_reconstruction_b{block_size}_h{hop_size}.wav"), y, sr)
