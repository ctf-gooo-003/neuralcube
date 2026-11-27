# 2026 - copyright - all rights reserved - clayton thomas baber

import os
import torch
import numpy as np
from time import time
from tqdm import tqdm
from cube import Cube
from model import RubikDistancePredictor, RubikEnsemble, load_eval_model
from torch import tensor, argsort, float32
from tkinter import Tk, filedialog
from collections import defaultdict


if __name__ == "__main__":
  root = Tk()
  root.withdraw()
  root.attributes("-topmost", True)

  eval_device = "cuda" if torch.cuda.is_available() else "cpu"
  model_paths = []
  all_endpoints = []

  num_models = int(input("number of models? "))
  for i in range(num_models):
    selected_file = filedialog.askopenfilename(
        initialdir="checkpoints",
        title=f"Select Model File {i + 1} of {num_models} (.ckpt or .pt)",
        filetypes=[
            ("Model Checkpoint or Shrunk", "*.ckpt *.pt"),
            ("Lightning Checkpoints", "*.ckpt"),
            ("Shrunk PyTorch Models", "*.pt"),
            ("All Files", "*.*")
        ]
    )

    if not selected_file:
      print(f"No file selected for model {i + 1}. Skipping.")
      continue

    model_paths.append(selected_file)
    run_dir = os.path.dirname(selected_file)

    # Load Endpoints (.dat binary memmap produced by dataset.py)
    endpoints_path = os.path.join(run_dir, "train_endpoints.dat")
    if os.path.exists(endpoints_path):
      file_size = os.path.getsize(endpoints_path)
      num_endpoints = file_size // 54
      run_endpoints = np.memmap(endpoints_path, dtype="uint8", mode="r", shape=(num_endpoints, 54))
      all_endpoints.append(np.array(run_endpoints))
    else:
      print(f"Warning: 'train_endpoints.dat' not found in {run_dir}")

  if not model_paths:
    raise FileNotFoundError("No model files selected.")

  if not all_endpoints:
    raise FileNotFoundError("No train_endpoints.dat found across selected model directories.")

  # Concatenate arrays across selected runs, deduplicate, and convert to set
  concatenated_endpoints = np.vstack(all_endpoints)
  cases = set(tuple(ep) for ep in concatenated_endpoints)

  model = RubikEnsemble(model_paths, device=eval_device)

  cube = Cube()
  total = len(cases)
  spacer = len(str(total))

  outcomes = {"success": 0, "dead": 0, "timeout": 0, "saved": 0}
  memo = {}

  start_time = time()
  with tqdm(total=total, desc="Evaluating") as pbar:
    for checked, case in enumerate(cases, start=1):
      cube.setState(case)
      history = defaultdict(int)

      outcome = None
      for i in range(42):
        state = cube.getState()
        probe = tensor(cube.getProbe(), dtype=float32)

        predictions = model(probe).squeeze()
        choices = argsort(predictions).cpu().numpy()

        choice = history[state]

        if choice < 18:
          action = int(choices[choice])
          history[state] += 1
        else:
          outcome = "dead"
          break

        cube.act(action)
        state = cube.getState()
        if state in memo:
          outcomes["saved"] += 1
          outcome = memo[state]
          break

        if cube.isSolved():
          outcome = "success"
          break

      if outcome is None:
        outcome = "timeout"

      outcomes[outcome] += 1
      for visited in history:
        memo[visited] = outcome

      pbar.set_postfix(
          accu=f"{outcomes['success']/checked:6.4f}",
          dead=f"{outcomes['dead']:0{spacer}d}",
          save=f"{outcomes['saved']:0{spacer}d}",
          solv=f"{outcomes['success']:0{spacer}d}",
          tout=f"{outcomes['timeout']:0{spacer}d}"
      )
      pbar.update(1)

  end_time = time()
  print(
      f"done evaluating: {outcomes['success']/total:.4f} accuracy. finished in {end_time - start_time:.2f}s\n",
      "\n".join(model_paths)
  )
