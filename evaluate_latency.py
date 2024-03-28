from pathlib import Path

import librosa as li
import numpy as np
from effortless_config import Config
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


def compute_latency(original, reconstruction, plot=False):
    """
    Compute the latency between the original and reconstructed audio.
    Turn plot to True to get a visual representation of the alignment.
    """
    # Create energy envelope
    # https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_NoveltyEnergy.html
    x_hat = np.convolve(reconstruction**2, np.hanning(2048)**2, mode="same")
    x = np.convolve(original**2, np.hanning(2048)**2, mode="same")

    # Look for -2000 to 5000 samples delays
    x_hat = np.pad(x_hat, (2000, 5000), mode="constant")
    corr = np.correlate(x_hat, x, mode="valid")
    #corr = corr[len(corr) // 2:]
    delay = np.argmax(corr)

    if plot:
        # plot the original, reconstruction and correlation on subplots
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))

        ax[0].plot(x, label="x")
        ax[0].plot(x_hat[2000:], label="x_hat")
        ax[0].set_title("Unaligned")
        ax[0].legend()

        ax[1].plot(x, label="x")
        ax[1].plot(x_hat[delay:], label="x_hat")
        ax[1].set_title(f"Aligned: Shifted {delay - 2000} samples")
        ax[1].legend()

        ax[2].plot(original, label="x")
        start = delay - 2000
        if start < 0:
            start = 0
            reconstruction = np.pad(reconstruction, (-start, 0), mode="constant")
        else:
            reconstruction = reconstruction[start:]
        ax[2].plot(reconstruction, label="x_hat")
        ax[2].set_title("Aligned Audio")
        ax[2].legend()

        ax[3].plot(np.arange(-2000, -2000 + len(corr)), corr)
        ax[3].set_title("Correlation")
        ax[3].axvline(delay - 2000, color="red")

        fig.tight_layout()

        plt.show()

    return delay - 2000

if __name__ == "__main__":

    class args(Config):
        DATASET = None
        RENDERS = None
    
    args.parse_args()

    # SEARCH FOR WAV FILES
    dataset = list(Path(args.DATASET).rglob("*.wav"))
    renders = list(Path(args.RENDERS).rglob("*.wav"))

    # Create map of audio ground truth and renders
    eval_map = {v.stem: {"original": v} for v in dataset}

    for render in renders:
        parts = render.stem.split("_")
        key = f"{parts[0]}_{parts[1]}"
        if len(parts) == 3:
            eval_map[key]["full"] = render
        elif len(parts) == 4:
            eval_map[key][f"full_{parts[3]}"] = render
        else:
            pass
            #eval_map[key][f"{parts[3]}_{parts[4]}"] = render
    
    # Evaluate
    results = {}
    for idx, audio in tqdm(eval_map.items()):
        original, sr = li.load(audio["original"], sr=None)
        energy = np.sum(np.square(original))
        if energy < 1e-6:
            continue

        for k, v in audio.items():
            if k == "original":
                continue
            reconstruction, sr2 = li.load(v, sr=None)
            assert sr == sr2
            assert original.shape == reconstruction.shape

            delay = compute_latency(original, reconstruction)
            if k not in results:
                results[k] = []

            results[k].append(delay)
    
    plt.boxplot(results.values(), labels=results.keys())
    plt.show()

    # Compute mean
    for k, dists in results.items():
        print(f"Mean Latency {k}: {np.mean(dists)} +/- {np.std(dists)}")
        print(f"Median Latency {k}: {np.median(dists)}")
