# Assignment 3 — MNIST CNN (28×28 Grayscale)

## Requirement coverage
- **Dataset:** MNIST 28×28 grayscale (70k images, 10 classes), loaded via `fetch_openml('mnist_784')`, split 60k train / 10k test, normalized 0–1, reshaped to N×1×28×28.
- **Architecture (exact):**
  - Conv1: 32 filters, 3×3, ReLU → MaxPool 2×2
  - Conv2: 64 filters, 3×3, ReLU → MaxPool 2×2
  - Flatten (64*5*5=1600) → FC 128 ReLU → Output 10 (Softmax via CrossEntropyLoss)
- **Training:** CrossEntropyLoss, Adam (lr=0.001), Epochs=10 (in 10–20 range), Batch=64, track loss+accuracy per epoch.
- **Evaluation:** Test accuracy + plots (Loss vs epochs, Accuracy vs epochs, Predicted vs Actual samples).

## Files
- `Assignment_3_MNIST_CNN.ipynb` — main submission notebook with comments + outputs
- `assignment3_cnn_mnist.py` — standalone runnable script (same logic)
- `mnist_cnn.pt` — trained weights
- `loss_vs_epochs.png`, `accuracy_vs_epochs.png`, `sample_predictions.png` — generated plots

## Results (actual run, CPU, seed 42)
| Epoch | Loss | Train Acc |
|---|---|---|
|1|0.1972|94.06%|
|2|0.0559|98.30%|
|3|0.0382|98.81%|
|4|0.0275|99.11%|
|5|0.0216|99.29%|
|6|0.0178|99.44%|
|7|0.0123|99.58%|
|8|0.0120|99.59%|
|9|0.0089|99.70%|
|10|0.0069|99.76%|
- **Test Accuracy: 98.92% (9892/10000)**

## How to run
```bash
pip install torch scikit-learn matplotlib
python assignment3_cnn_mnist.py
# or open Assignment_3_MNIST_CNN.ipynb in Jupyter and Run All
```
Notebook needs internet on first run to download MNIST via OpenML (cached afterwards).
No `torchvision` required.
