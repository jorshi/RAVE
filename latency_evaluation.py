"""
Script to evaluate latency on a trained RAVE model
"""

import argparse
import logging
import sys

import cached_conv as cc
import gin
from nleval import NeuralLatencyModelWrapper
from nleval import NeuralLatencyEvaluator
import rave
import torch


class LatencyModelWrapper(NeuralLatencyModelWrapper):

    def __init__(self, model: rave.RAVE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of the model.
        """
        return self.model.forward(x)

    def reset(self) -> None:
        """
        Reset the model state.
        """
        # Reset the internal cache of the model by passing zeros through the model
        # TODO: can we directly update the cache and would that be better?
        with torch.no_grad():
            x = torch.zeros(1, 1, self.samplerate() * 2, device=self.current_device())
            self.forward(x)

    def samplerate(self) -> int:
        """
        Return the samplerate of the model.
        """
        return self.model.sr

    def current_device(self) -> torch.device:
        """
        Return the device the model is on.
        """
        if hasattr(self, "device"):
            return self.device
        else:
            raise RuntimeError("Model has not been moved to a device")

    def to(self, device: torch.device) -> None:
        """
        Move the model to a device.
        """
        self.model.to(device)
        self.device = device


def load_model(model_path):
    config_path = rave.core.search_for_config(model_path)
    if config_path is None:
        logging.error("config not found in folder %s" % model_path)
    gin.parse_config_file(config_path)
    model = rave.RAVE()
    run = rave.core.search_for_run(model_path)
    if run is None:
        logging.error("run not found in folder %s" % model_path)
    model = model.load_from_checkpoint(run)
    model = LatencyModelWrapper(model).eval()
    return model


def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model_path", help="Path to model checkpoint and config", type=str
    )
    parser.add_argument(
        "name", help="Name of the eval run", type=str
    )
    parser.add_argument("--stream", help="Simulate streaming mode", action="store_true")
    parser.add_argument(
        "--gpu", help="Chunk size for streaming mode", type=int, default=-1
    )

    args = parser.parse_args(arguments)

    torch.set_float32_matmul_precision("high")
    cc.use_cached_conv(args.stream)

    # Create a latency evaluation model from the loaded RAVE model
    model = load_model(args.model_path)

    # device
    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    # Send the model to the device
    model.to(device)

    # Create the evaluator
    evaluator = NeuralLatencyEvaluator(model, args.name)
    evaluator.evaluate()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
