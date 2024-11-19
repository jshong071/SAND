# shape of the tensor passed across all modules are kept as (batch, d, seq_len)
import torch
from torch.optim import Adam
from DataUtils import getData, DataLoader
from TransformerModel import Transformer, GetImputation

### Defining the device
device = torch.device("cpu")

### Defining data
data_name = "HighDim_E"
sparsity = "dense"
error = False

### Settings for dataloader
dataloader_settings = {"d": 120,
                       "batch_size": 64, 
                       "split": (90, 5, 5), 
                       "device": device,
                       "data_name": data_name}

### Settings for Transformer and SAND
optimizer_settings = {"save_model_every": 200,
                      "lr": 3e-4,
                      "weight_decay": 1e-8}

model_settings = {"num_heads": 2,
                  "num_layers": (6, 6),
                  "num_hiddens": 128,
                  "max_epoch": 5000 + 1, 
                  "dropout": 0.15,
                  "batch_size": dataloader_settings["batch_size"],
                  "f_in": (2 + dataloader_settings["d"], 1 + dataloader_settings["d"]),
                  "device": device,
                  "dataloader_settings": dataloader_settings}

### Getting data
X, T, model_settings["sparsity_error_folder"] = getData(data_name = data_name, sparsity = sparsity, error = error)

### dataloader setting
dataLoader = DataLoader(X, T, dataloader_settings)
model_settings.update({ # Store dataset-specific scaling and time points for training
    "min_X": dataLoader.min_X,
    "max_X": dataLoader.max_X,
    "tptsTraining": dataLoader.tptsTraining
})

### Training
model = Transformer(model_settings).to(device)
optimizer = Adam(model.parameters(), lr = optimizer_settings["lr"], weight_decay = optimizer_settings["weight_decay"])
model.StartTraining(dataLoader, optimizer, optimizer_settings["save_model_every"], verbose = True)

### Getting impuations on the testing batch
X_test = X[int(X.shape[0] * (sum(dataloader_settings["split"][0:2])/sum(dataloader_settings["split"]))):]
T_test = T[int(X.shape[0] * (sum(dataloader_settings["split"][0:2])/sum(dataloader_settings["split"]))):]
SAND_imp, VT_imp = GetImputation(X_test, T_test, data_name = "HighDim_E", sparsity = "dense", error = False)
