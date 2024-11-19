# Import required libraries
# The tensor shape across all modules is maintained as (batch, d, seq_len)
import torch
from torch.optim import Adam
from DataUtils import getData, DataLoader  # Custom utility for data preparation
from TransformerModel import Transformer, GetImputation  # Custom transformer and imputation methods

# Define the computational device (CPU/GPU)
# Use GPU if available, fallback to CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-specific configuration
# Specify the dataset name, sparsity type, and whether to include error terms
data_name = "HighDim_E"  # Dataset name
sparsity = "dense"       # Sparsity type: "dense" or "sparse" or "ssparse"
error = False            # Include error in data (True/False)

# DataLoader configuration
# Configure feature dimensionality, batch size, train-validation-test split, and device
dataloader_settings = {
    "d": 120,                    # Dimensionality of input features
    "batch_size": 64,            # Number of samples per batch
    "split": (90, 5, 5),         # Train/Validation/Test split (in percentages)
    "device": device,            # Computational device
    "data_name": data_name       # Dataset name
}

# Transformer and optimization configuration
# Specify optimizer settings and Transformer model parameters
optimizer_settings = {
    "save_model_every": 200,     # Save model every 200 epochs
    "lr": 3e-4,                  # Learning rate
    "weight_decay": 1e-8         # Weight decay (regularization)
}

model_settings = {
    "num_heads": 2,                     # Number of attention heads
    "num_layers": (6, 6),               # Number of encoder and decoder layers
    "num_hiddens": 128,                 # Hidden layer size
    "max_epoch": 5000,                  # Maximum number of training epochs
    "dropout": 0.15,                    # Dropout rate for regularization
    "batch_size": dataloader_settings["batch_size"],  # Batch size
    "f_in": (2 + dataloader_settings["d"], 1 + dataloader_settings["d"]),  # Input dimensions
    "device": device,                   # Computational device
    "dataloader_settings": dataloader_settings  # DataLoader configuration
}

# Load the dataset
# Retrieves data matrices X (features) and T (timestamps) along with folder configuration
X, T, model_settings["sparsity_error_folder"] = getData(
    data_name=data_name,
    sparsity=sparsity,
    error=error
)

# Initialize the DataLoader
# Splits data into train/validation/test and normalizes it for training
dataLoader = DataLoader(X, T, dataloader_settings)

# Update model settings with dataset-specific parameters
model_settings.update({
    "min_X": dataLoader.min_X,           # Minimum value of the dataset (for normalization)
    "max_X": dataLoader.max_X,           # Maximum value of the dataset (for normalization)
    "tptsTraining": dataLoader.tptsTraining  # Training time points
})

# Initialize the Transformer model and optimizer
# The Transformer is augmented with the SAND module for smooth imputation
model = Transformer(model_settings).to(device)
optimizer = Adam(
    model.parameters(),
    lr=optimizer_settings["lr"],
    weight_decay=optimizer_settings["weight_decay"]
)

# Start model training
# Trains the Transformer on the provided data and saves intermediate checkpoints
model.StartTraining(
    dataLoader,
    optimizer,
    optimizer_settings["save_model_every"],
    verbose=True  # Set verbose=True to display training progress
)

# Prepare test data for evaluation
# Extracts the test set from the data matrix using the specified split
test_split_ratio = sum(dataloader_settings["split"][0:2]) / sum(dataloader_settings["split"])
X_test = X[int(X.shape[0] * test_split_ratio):]
T_test = T[int(X.shape[0] * test_split_ratio):]

# Perform imputation on the testing batch
# Uses the trained Transformer with SAND for imputing missing values
SAND_imp, VT_imp = GetImputation(
    X_test,
    T_test,
    data_name=data_name,
    sparsity=sparsity,
    error=error
)

# Notes:
# - SAND_imp contains the imputed values using the SAND module.
# - VT_imp contains imputed values using the vanilla Transformer.
