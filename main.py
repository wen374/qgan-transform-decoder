import model
import simulate
import dataset

import torch
from torch.utils.data import Dataset, DataLoader

def main():
    print("generating noise model")
    noise_model = simulate.get_noise(0.01, 0.01)

    # Create a dataset
    print("creating dataset")
    ds = dataset.SurfaceCodeDataset(noise_model)

    # Create a model
    print("creating model")
    nn = model.SurfaceCodeDecoder(
        n_attn_dims=8,
        n_heads=4,
        n_attn_layers=2,
        n_ff_layers=2,
        n_ff_dims=64,
        dropout=0.1,
        max_seq_len=511,
    )
    
    # Dataloader
    print("Creating DataLoader")
    dl = DataLoader(ds, batch_size=16, num_workers=2)

    # Train the model
    print("Training model")
    nn.train()

    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(10):
        loss = 0
        for batch in dl:
            optimizer.zero_grad()

            syndrome, logical, logical_state = batch
            output = nn(syndrome, logical)

            loss = loss_fn(output, logical_state.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            loss += loss.item()

        print('loss:', loss.item())

    # Test the model
    print("Testing model")
    nn.eval()
    with torch.no_grad():
        for batch in dl:
            syndrome, logical, logical_state = batch
            output = nn(syndrome, logical)

            print(output)
            print(logical_state)

            break



    


if __name__ == '__main__':
    main()