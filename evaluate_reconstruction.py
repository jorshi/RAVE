from pathlib import Path

import librosa as li
import numpy as np
from effortless_config import Config
import torch
from tqdm import tqdm

def signal_to_distortion(original, reconstruction, eps=1e-7):
    """
    https://www.frontiersin.org/articles/10.3389/frsip.2021.808395/full
    """
    signal = np.sum(np.square(original)) + eps
    distortion = np.sum(np.square(original - reconstruction)) + eps
    return 10 * np.log10(distortion / signal)


def _log_magnitude_stft(x: torch.Tensor, n_fft=2048, hop_size=64, eps=1e-8):
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_size,
        window=torch.hann_window(n_fft, device=x.device),
        return_complex=True,
    )
    return torch.log(torch.square(torch.abs(X)) + eps)


def log_spectral_distance(original, reconstruction):
    """
    """
    x = torch.from_numpy(original).float()
    y = torch.from_numpy(reconstruction).float()
    X = _log_magnitude_stft(x)
    Y = _log_magnitude_stft(y)

    # Mean of the squared difference along the frequency axis
    lsd = torch.mean(torch.square(X - Y), dim=-2)

    # Mean of the square root over the temporal axis
    lsd = torch.mean(torch.sqrt(lsd), dim=-1)

    return lsd


if __name__ == "__main__":

    class args(Config):
        FOLDER = "reconstruction/"
    
    args.parse_args()

    # SEARCH FOR WAV FILES
    audios = list(Path(args.FOLDER).rglob("*.wav"))

    # Create map of audio
    eval_map = {}
    for audio in audios:
        parts = audio.stem.split("_")
        idx = int(parts[1])
        if idx not in eval_map:
            eval_map[idx] = {}
        
        if len(parts) == 4:
            if parts[3] == "cached":
                eval_map[idx]["original"] = audio
        else:
            eval_map[idx][f"{parts[3]}_{parts[4]}"] = audio

    # Evaluate
    sdr = {}
    for idx, audio in tqdm(eval_map.items()):
        original, sr = li.load(audio["original"], sr=None)
        for k, v in audio.items():
            if k == "original":
                continue
            reconstruction, sr2 = li.load(v, sr=None)
            assert sr == sr2
            assert original.shape == reconstruction.shape

            # Compute distortion
            distortion = log_spectral_distance(original, reconstruction)

            if k not in sdr:
                sdr[k] = []
            sdr[k].append(distortion)
    
    # Compute mean
    for k, dists in sdr.items():
        print(f"Buffer Size {k}: {np.mean(dists)} +/- {np.std(dists)}")
