# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]

import sys
import platform
if platform.machine().lower() in ("arm64", "aarch64"):
    print(
        "Warning: stim is not supported on manylinux ARM64/aarch64. Skipping this example..."
    )
    sys.exit(0)

import stim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --------------------------
# Parameters
# --------------------------
distance = 3  # Surface code distance (simpler for demo)
num_rounds = 3  # Rounds of stabilizer measurements
num_train_samples = 5000  # Training samples (more data)
num_val_samples = 1000  # Validation samples
num_test_samples = 1000  # Test samples
hidden_dim = 128  # Larger model capacity
error_prob = 0.005  # Balanced error rate for better learning

# --------------------------
# Build the surface code circuit
# --------------------------
# Use the built-in Stim surface code generator with noise
circuit = stim.Circuit.generated("surface_code:rotated_memory_x",
                                 distance=distance,
                                 rounds=num_rounds,
                                 after_clifford_depolarization=error_prob,
                                 after_reset_flip_probability=error_prob,
                                 before_measure_flip_probability=error_prob,
                                 before_round_data_depolarization=error_prob)

# Convert to detector error model
dem = circuit.detector_error_model()
with open("surface_code_decoder.dem", "w") as dem_file:
    dem_file.write(str(dem))
num_detectors = dem.num_detectors
num_data_qubits = circuit.num_qubits - num_detectors

print(f"Num data qubits: {num_data_qubits}, Num detectors: {num_detectors}")

# --------------------------
# Sample training, validation, and test data
# --------------------------
sampler = circuit.compile_detector_sampler()


def sample_data(num_samples):
    """Sample detector outcomes and observable flips."""
    X_data = []
    Y_data = []

    detector_samples, observable_samples = sampler.sample(
        num_samples, separate_observables=True)

    for i in range(num_samples):
        detectors = torch.tensor(detector_samples[i], dtype=torch.float32)
        observable = torch.tensor(observable_samples[i], dtype=torch.float32)
        X_data.append(detectors)
        Y_data.append(observable)

    return torch.stack(X_data), torch.stack(Y_data)


print(f"Sampling {num_train_samples} training samples...")
X_train, Y_train = sample_data(num_train_samples)

print(f"Sampling {num_val_samples} validation samples...")
X_val, Y_val = sample_data(num_val_samples)

print(f"Sampling {num_test_samples} test samples...")
X_test, Y_test = sample_data(num_test_samples)

num_observables = Y_train.shape[1]
print(f"Num observables: {num_observables}")


# --------------------------
# Improved Torch NN decoder with dropout and deeper architecture
# --------------------------
class SurfaceCodeDecoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 128
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


model = SurfaceCodeDecoder(input_dim=num_detectors,
                           output_dim=num_observables,
                           hidden_dim=hidden_dim,
                           dropout=0.3)
optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Lower learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=20)
criterion = nn.BCELoss()

# Create DataLoaders for batch training
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


def compute_accuracy(predictions, targets, threshold=0.5):
    """Compute binary accuracy."""
    pred_binary = (predictions > threshold).float()
    correct = (pred_binary == targets).float().mean()
    return correct.item()


# --------------------------
# Train NN with validation
# --------------------------
epochs = 1000  # Train longer for better convergence
best_val_acc = 0.0
print("\nTraining started...")
print("=" * 70)

for epoch in range(epochs):
    # Training step with batches
    model.train()
    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        train_output = model(batch_X)
        train_loss = criterion(train_output, batch_Y)
        train_loss.backward()
        optimizer.step()

        train_loss_total += train_loss.item() * batch_X.size(0)
        train_correct += ((train_output > 0.5).float() == batch_Y).sum().item()
        train_total += batch_Y.numel()

    train_loss_avg = train_loss_total / len(train_loader.dataset)
    train_acc = train_correct / train_total

    # Validation step with batches
    model.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    cum_ler = 0.0

    with torch.no_grad():

        for batch_X, batch_Y in val_loader:
            val_output = model(batch_X)
            val_loss = criterion(val_output, batch_Y)

            val_output_binary = (val_output > 0.5)
            ler = val_output_binary ^ (batch_Y > 0.5)
            # print(f"loss: {loss.sum().item() / loss.numel()}")
            cum_ler += ler.sum().item()
            # print(f"val_output_binary: {val_output_binary} ler: {ler} batch_Y: {batch_Y} ")

            # print(f"logical_error_rate (pred): {val_output.sum().item() / val_output.numel()}")
            # print(f"logical_error_rate (raw): {batch_Y.sum().item() / batch_Y.numel()}")

            val_loss_total += val_loss.item() * batch_X.size(0)
            val_correct += ((val_output > 0.5).float() == batch_Y).sum().item()
            val_total += batch_Y.numel()

    # print(f"logical_error_rate (raw): {batch_Y.sum().item() / batch_Y.numel()}")
    # print(f"cum_ler: {cum_ler / len(val_loader.dataset)}")

    val_loss_avg = val_loss_total / len(val_loader.dataset)
    val_acc = val_correct / val_total

    # Learning rate scheduling
    scheduler.step(val_loss_avg)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "surface_code_decoder_best.pth")

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")

print("=" * 70)
print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")

# --------------------------
# Load best model and evaluate on test set
# --------------------------
print("\nLoading best model and evaluating on test set...")
model.load_state_dict(torch.load("surface_code_decoder_best.pth"))
model.eval()

with torch.no_grad():
    test_output = model(X_test)
    test_loss = criterion(test_output, Y_test)
    test_acc = compute_accuracy(test_output, Y_test)

    # Additional metrics
    test_pred_binary = (test_output > 0.5).float()

    # Count logical errors
    total_actual_errors = Y_test.sum().item()
    total_predicted_errors = test_pred_binary.sum().item()
    correct_predictions = (test_pred_binary == Y_test).sum().item()
    total_predictions = Y_test.numel()

print("=" * 70)
print("TEST SET RESULTS:")
print("=" * 70)
print(f"Test Loss:                    {test_loss.item():.4f}")
print(f"Test Accuracy:                {test_acc:.4f} ({test_acc*100:.2f}%)")
print(
    f"Correct predictions:          {correct_predictions}/{total_predictions}")
print(f"Actual logical errors:        {int(total_actual_errors)}")
print(f"Predicted logical errors:     {int(total_predicted_errors)}")
print("=" * 70)

# --------------------------
# Export to ONNX
# --------------------------
print("\nExporting model to ONNX...")
torch.onnx.export(model,
                  X_train[:1],
                  "surface_code_decoder.onnx",
                  input_names=["detectors"],
                  output_names=["observable_probs"],
                  opset_version=17)
print("ONNX model saved as surface_code_decoder.onnx")
print("Detector error model saved as surface_code_decoder.dem")
print("PyTorch weights saved as surface_code_decoder_best.pth")
