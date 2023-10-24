# Assignment3: GPT

## Task
For this task, you're required to finalize the code that trains a toy GPT model on a toy number sorting dataset.
While the training framework and supplementary functions have been pre-established, your primary responsibility is to complete the sections demarcated by:

```python
# --- TODO: start of your code ---

# --- TODO: end of your code ---
```

This ensures the model functions correctly.
It's advisable not to modify any part of the code outside of the `TODO` block.
However, if you notice the model's training process is slow (which it typically shouldn't), you may choose to create your own functions for saving/loading the model during training and evaluation.
**Please keep the TODO comments in your submission**.

Specifically, you'll focus on the following functions:
- `src.model.GELU.forward`: Implement the GELU activation function, a standard in many pre-trained networks.
- `src.CausalSelfAttention.forward`: Define the forward pass for the multi-head masked self-attention layer.
- `src.GPT.inference`: Set up the inference method for the GPT model.

For a comprehensive understanding and specifications, please check the annotations present before each `TODO` section within the respective functions.

## Environment Setup
The code is built with Python 3.10.
Other package requirements are listed in `requirements.txt`.
You are suggested to run the code in an isolated virtual [conda](https://www.anaconda.com/) environment.
Suppose you have already install conda in your device, you can create a new environment and activate it with
```bash
conda create -n 310 python=3.10
conda activate 310
```
Then, you can install the required packages with
```bash
pip install -r requirements.txt
```

Alternatively, you can also use other Python version manager or virtual environments such as [pyenv](https://github.com/pyenv/pyenv) or [docker](https://www.docker.com/) to you prefer.

## Run

You can also run the code with 
```bash
[CUDA_VISIBLE_DEVICES=...] python run.py --name <your name> --gtid <your GTID> [other arguments...]

# for example, python run.py --name "George Burdell" --gtid 123456789 --lr 1e5
```

## Submission

If your code runs successfully, you will see a `record.log` file in your `log` folder suppose you keep the `--log_path` argument as default.
The log file should track your training status and report the training loss and the final test performance.
If your code is correct, the test performance should range from 95% to 100% when the model is trained for 2000 iterations.
You may also fine-tune the hyperparameters to achieve better performance.

For this assignment, you should submit a `ner.<GivenName>.<FamilyName>.<GTID>.zip` (e.g. `ner.George.Burdell.901234567.zip`) file containing `./src/` and `./log/` folders and all their contents.
**Do not include `./data/`, `.gitignore`, `LICENSE` or other files or folders.**
If your using Unix-like systems, you can run
```bash
zip -r ner.<GivenName>.<FamilyName>.<GTID>.zip log/ src/
```

## Reference
```
10/24/2023 18:23:00 - INFO - src.args -   Configurations:
{
  "batch_size": 64,
  "d_model": 128,
  "device": "cuda",
  "digit_seq_len": 6,
  "dropout": 0.1,
  "gpt_seq_len": 11,
  "grad_norm_clip": 1.0,
  "gtid": "901234567",
  "log_path": "log/record.log",
  "lr": 0.0005,
  "n_digits": 3,
  "n_head": 4,
  "n_layer": 4,
  "n_training_steps": 2000,
  "name": "George Burdell",
  "no_cuda": false,
  "num_workers": 0,
  "seed": 42
}
10/24/2023 18:23:00 - INFO - __main__ -   name: George Burdell
10/24/2023 18:23:00 - INFO - __main__ -   GTID: 901234567
10/24/2023 18:23:00 - INFO - __main__ -   Peaking at a training example:
10/24/2023 18:23:00 - INFO - __main__ -   x = [0, 2, 1, 1, 0, 2, 0, 0, 1, 1, 2]
10/24/2023 18:23:00 - INFO - __main__ -   y = [-1, -1, -1, -1, -1, 0, 0, 1, 1, 2, 2]
10/24/2023 18:23:00 - INFO - src.model -   number of parameters: 0.80M
10/24/2023 18:23:01 - INFO - src.trainer -   running on device cuda
10/24/2023 18:23:02 - INFO - src.trainer -   iter_dt 0.00ms; iter 0: train loss 1.09242
10/24/2023 18:23:05 - INFO - src.trainer -   iter_dt 20.87ms; iter 100: train loss 0.02907
10/24/2023 18:23:07 - INFO - src.trainer -   iter_dt 21.35ms; iter 200: train loss 0.01861
10/24/2023 18:23:09 - INFO - src.trainer -   iter_dt 23.39ms; iter 300: train loss 0.01358
10/24/2023 18:23:11 - INFO - src.trainer -   iter_dt 23.93ms; iter 400: train loss 0.01688
10/24/2023 18:23:14 - INFO - src.trainer -   iter_dt 21.48ms; iter 500: train loss 0.04482
10/24/2023 18:23:16 - INFO - src.trainer -   iter_dt 21.28ms; iter 600: train loss 0.00185
10/24/2023 18:23:18 - INFO - src.trainer -   iter_dt 23.81ms; iter 700: train loss 0.01794
10/24/2023 18:23:20 - INFO - src.trainer -   iter_dt 20.55ms; iter 800: train loss 0.08734
10/24/2023 18:23:23 - INFO - src.trainer -   iter_dt 23.25ms; iter 900: train loss 0.00234
10/24/2023 18:23:25 - INFO - src.trainer -   iter_dt 20.70ms; iter 1000: train loss 0.01730
10/24/2023 18:23:27 - INFO - src.trainer -   iter_dt 25.71ms; iter 1100: train loss 0.00975
10/24/2023 18:23:29 - INFO - src.trainer -   iter_dt 19.87ms; iter 1200: train loss 0.00818
10/24/2023 18:23:31 - INFO - src.trainer -   iter_dt 20.13ms; iter 1300: train loss 0.00747
10/24/2023 18:23:33 - INFO - src.trainer -   iter_dt 20.03ms; iter 1400: train loss 0.05385
10/24/2023 18:23:36 - INFO - src.trainer -   iter_dt 19.97ms; iter 1500: train loss 0.00072
10/24/2023 18:23:38 - INFO - src.trainer -   iter_dt 20.25ms; iter 1600: train loss 0.00301
10/24/2023 18:23:40 - INFO - src.trainer -   iter_dt 19.82ms; iter 1700: train loss 0.00232
10/24/2023 18:23:42 - INFO - src.trainer -   iter_dt 19.82ms; iter 1800: train loss 0.02103
10/24/2023 18:23:44 - INFO - src.trainer -   iter_dt 19.87ms; iter 1900: train loss 0.00189
10/24/2023 18:23:46 - INFO - __main__ -   Evaluating on train set...
10/24/2023 18:23:52 - INFO - src.trainer -   final score: 5000/5000 = 100.00% correct
10/24/2023 18:23:52 - INFO - __main__ -   Evaluating on test set...
10/24/2023 18:23:53 - INFO - src.trainer -   final score: 5000/5000 = 100.00% correct
10/24/2023 18:23:53 - INFO - __main__ -   input sequence   :[[0, 0, 2, 1, 0, 1]]
10/24/2023 18:23:53 - INFO - __main__ -   predicted sorted :[[0, 0, 0, 1, 1, 2]]
10/24/2023 18:23:53 - INFO - __main__ -   gt sort          :[0, 0, 0, 1, 1, 2]
10/24/2023 18:23:53 - INFO - __main__ -   matches          :True
```
