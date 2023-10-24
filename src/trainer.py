"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import logging
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        self.device = config.device
        self.model = self.model.to(self.device)
        logger.info(f"running on device {self.device}")

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            _, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.on_batch_end()

            self.iter_num += 1
            t_now = time.time()
            self.iter_dt = t_now - self.iter_time
            self.iter_time = t_now

            # termination conditions
            if self.iter_num >= config.n_training_steps:
                break

    def on_batch_end(self):
        if self.iter_num % 100 == 0:
            logger.info(f"iter_dt {self.iter_dt * 1000:.2f}ms; iter {self.iter_num}: train loss {self.loss.item():.5f}")

    def eval_split(self, dataset, max_batches):
        self.model.eval()

        results = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        for b, (x, y) in enumerate(loader):
            x = x.to(self.device)
            y = y.to(self.device)
            # isolate the input pattern alone
            inputs = x[:, : self.config.digit_seq_len]
            solution = y[:, -self.config.digit_seq_len :]
            # let the model sample the rest of the sequence
            cat = self.model.inference(inputs, self.config.digit_seq_len)  # using greedy argmax, not sampling
            sol_candidate = cat[:, self.config.digit_seq_len :]  # isolate the filled in sequence
            # compare the predicted sequence to the true sequence
            correct = (solution == sol_candidate).all(1).cpu()
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 3:  # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    logger.warning(
                        "GPT claims that %s sorted is %s but gt is %s"
                        % (inputs[i].tolist(), sol_candidate[i].tolist(), solution[i].tolist())
                    )
            if max_batches is not None and b + 1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        logger.info("final score: %d/%d = %.2f%% correct" % (rt.sum(), len(results), 100 * rt.mean()))
        return rt.sum()
