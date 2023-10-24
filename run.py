import os.path as osp
import sys
import torch
import logging
from transformers import HfArgumentParser, set_seed
from src.dataset import SortDataset
from src.io import set_logging
from src.args import Arguments, Config
from src.trainer import Trainer
from src.model import GPT

logger = logging.getLogger(__name__)


def main(args):
    config = Config().from_args(args).log()

    assert config.name is not None, f"Student name is not specified!"
    assert config.gtid is not None, f"Student GTID is not specified!"
    logger.info(f"name: {config.name}")
    logger.info(f"GTID: {config.gtid}")

    # print an example instance of the dataset
    train_dataset = SortDataset("train", length=config.digit_seq_len, num_digits=config.n_digits)
    test_dataset = SortDataset("test", length=config.digit_seq_len, num_digits=config.n_digits)

    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", represented as -1, as the transformer is reading the input sequence
    """
    logger.info("Peaking at a training example:")
    x, y = train_dataset[0]
    logger.info(f"x = {x.tolist()}")
    logger.info(f"y = {y.tolist()}")

    model = GPT(config)

    # create a Trainer object
    trainer = Trainer(config, model, train_dataset)

    trainer.run()

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        logger.info("Evaluating on train set...")
        trainer.eval_split(train_dataset, max_batches=50)
        logger.info("Evaluating on test set...")
        trainer.eval_split(test_dataset, max_batches=50)

    # let's run a random given sequence through the model as well
    inputs = torch.tensor([[0, 0, 2, 1, 0, 1]], dtype=torch.long).to(trainer.device)
    solution = torch.sort(inputs[0])[0]
    assert inputs[0].nelement() == config.digit_seq_len
    with torch.no_grad():
        preds = model.inference(inputs, config.digit_seq_len)
    preds = preds[:, config.digit_seq_len :]

    logger.info(f"input sequence   :{inputs.tolist()}")
    logger.info(f"predicted sorted :{preds.tolist()}")
    logger.info(f"gt sort          :{solution.tolist()}")
    logger.info(f"matches          :{bool((solution == preds).all())}")


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    set_logging(log_path=arguments.log_path)
    set_seed(arguments.seed)

    main(args=arguments)
