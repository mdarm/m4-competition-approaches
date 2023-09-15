import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ESRNN
from utils import SequenceLabelingDataset, process_and_split_data
from evaluation_metrics import *

def h1_run():
    h1_train, h1_test = process_and_split_data("../data/Hourly-train.csv",
                                               "../data/Hourly-test.csv", 'H1')
    train = h1_train[:-48]
    test = h1_train

    sl = SequenceLabelingDataset(train, len(train), False)
    sl_t = SequenceLabelingDataset(test, len(test), False)

    train_dl = DataLoader(dataset=sl, batch_size=512, shuffle=False)
    test_dl = DataLoader(dataset=sl_t, batch_size=512, shuffle=False)

    hw = ESRNN(hidden_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = hw.to(device)

    opti = torch.optim.Adam(hw.parameters(), lr=0.01)

    overall_loss_train = []
    overall_loss = []
    patience = 2 
    counter = 0
    best_loss = float('inf')

    for epoch in tqdm(range(20)):
        loss_list_b = []
        train_loss_list_b = []

        # Use batches of past and forecasted value.
        # Batches are determined by a random start integer.
        for batch in iter(train_dl):
            inp = batch[0].float().to(device)
            out = batch[1].float().to(device)
            shifts = batch[2].numpy()
            pred = hw(inp, shifts)
            loss = F.l1_loss(pred, out)
            train_loss_list_b.append(loss.detach().cpu().numpy())
            
            opti.zero_grad()
            loss.backward()
            opti.step()

        # Use all the available values to forecast the future ones and eval on it.
        for batch in iter(test_dl):
            inp = batch[0].float().to(device)
            out = batch[1].float().to(device)
            shifts = batch[2].numpy()
            pred = hw(inp, shifts)
            loss = F.l1_loss(pred, out)
            loss_list_b.append(loss.detach().cpu().numpy())

        val_loss = np.mean(loss_list_b)
        train_loss = np.mean(train_loss_list_b)
        print("Validation L1-loss: ", val_loss)
        print("Training L1-loss: ", train_loss)
        overall_loss.append(np.mean(loss_list_b))
        overall_loss_train.append(np.mean(train_loss_list_b))

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break


    # Plot of Train and Validation Loss
    plt.plot(overall_loss, "g", label="Validation L1-loss")
    plt.plot(overall_loss_train, "r", label="Training L1-loss")
    plt.legend()
    plt.savefig("../output/training-loss.png")
    plt.close()

    # Calculate Projections
    batch  = next(iter(test_dl))
    inp    = batch[0].float().to(device)
    out    = batch[1].float().to(device)
    shifts = batch[2].numpy()
    pred   = hw(torch.cat([inp, out], dim=1), shifts)
    
    # Fetch hourly-position of projections 
    start_x  = len(inp[0]) + len(out[0, :])
    x_values = np.arange(start_x, start_x + len(pred[-48,:]))

    # Visualise Projections
    plt.figure(figsize=(12, 6))
    plt.plot(torch.cat([inp[0], out[0,:]]).cpu().detach().numpy(),
             'g', label='Actual Values', linewidth=2)
    plt.plot(x_values, pred[-48,:].cpu().detach().numpy(),
             'r', label='Predicted Values', linewidth=2)

    plt.title('ES-RNN Forecasting for H1 hourly dataset.')
    plt.xlabel('Hours')
    plt.ylabel('Value')
    plt.legend()

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("../output/hourly-projections.png")
    plt.close()

    # Compute and print OWA of the final forecast
    owa(pred.detach().cpu().numpy()[-48:][0], np.asarray(h1_test), seasonality=24)


if __name__ == "__main__":
    h1_run()
