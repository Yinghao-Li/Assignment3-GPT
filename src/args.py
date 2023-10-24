import os.path as osp
import json
import torch
import logging
from functools import cached_property
from dataclasses import dataclass, field
from src.io import prettify_json

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- student arguments ---
    name: str = field(default=None, metadata={"help": "name of the student model"})
    gtid: str = field(default=None, metadata={"help": "GTID of the student"})

    # --- manage directories and IO ---
    n_layer: int = field(default=4, metadata={"help": "number of layers in the model"})
    n_head: int = field(default=4, metadata={"help": "number of heads in the model"})
    d_model: int = field(default=128, metadata={"help": "dimension of the model"})
    log_path: str = field(
        default=osp.join("log", "record.log"),
        metadata={"help": "Path to log directory"},
    )

    # --- training arguments ---
    lr: float = field(default=5e-4, metadata={"help": "learning rate"})
    batch_size: int = field(default=64, metadata={"help": "model training batch size"})
    n_training_steps: int = field(default=2000, metadata={"help": "number of denoising model training epochs"})
    grad_norm_clip: float = field(default=1.0, metadata={"help": "gradient norm clipping"})
    dropout: float = field(default=0.1, metadata={"help": "dropout rate"})
    seed: int = field(default=42, metadata={"help": "random seed"})
    num_workers: int = field(default=0, metadata={"help": "number of workers for data loading"})

    # --- device arguments ---
    no_cuda: bool = field(default=False, metadata={"help": "Disable CUDA even when it is available"})

    @cached_property
    def device(self) -> str:
        """
        The device used by this process.
        """
        if not self.no_cuda and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        return device


@dataclass
class Config(Arguments):
    n_digits = 3
    digit_seq_len = 6

    @cached_property
    def gpt_seq_len(self):
        """
        the length of the sequence that will feed into transformer,
        containing concatenated input and the output, but -1 because
        the transformer starts making predictions at the last input element
        """
        return self.digit_seq_len * 2 - 1

    def from_args(self, args):
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: BertConfig)
        """
        arg_elements = {
            attr: getattr(args, attr)
            for attr in dir(args)
            if not callable(getattr(args, attr)) and not attr.startswith("__") and not attr.startswith("_")
        }
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def log(self):
        """
        Log all configurations
        """
        elements = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not (attr.startswith("__") or attr.startswith("_"))
        }
        logger.info(f"Configurations:\n{prettify_json(json.dumps(elements, indent=2), collapse_level=2)}")

        return self
