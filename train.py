import argparse
import datetime
import logging
import os

import torch
from torch.utils.data import DataLoader

import models
import utils


def main():
    parser = argparse.ArgumentParser(description="Train C-SWM World Model")

    # Training Hyperparameters
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate.')

    # Model Hyperparameters
    parser.add_argument('--encoder', type=str, default='small', help='Object extractor CNN size (e.g., `small`).')
    parser.add_argument('--sigma', type=float, default=0.5, help='Energy scale.')
    parser.add_argument('--hinge', type=float, default=1.0, help='Hinge threshold parameter.')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Number of hidden units in transition MLP.')
    parser.add_argument('--embedding-dim', type=int, default=2, help='Dimensionality of embedding.')
    parser.add_argument('--action-dim', type=int, default=4, help='Dimensionality of action space.')
    parser.add_argument('--num-objects', type=int, default=5, help='Number of object slots in model.')
    parser.add_argument('--ignore-action', action='store_true', default=False, help='Ignore action in GNN transition model.')
    parser.add_argument('--copy-action', action='store_true', default=False, help='Apply same action to all object slots.')

    # Setup & Logging
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--log-interval', type=int, default=20, help='How many batches to wait before logging.')
    parser.add_argument('--dataset', type=str, default='data/shapes_train.h5', help='Path to replay buffer.')
    parser.add_argument('--name', type=str, default='none', help='Experiment name.')
    parser.add_argument('--save-folder', type=str, default='checkpoints', help='Path to save checkpoints.')

    args = parser.parse_args()

    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Seed initialization
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Output Directory & Logging Setup
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = timestamp if args.name == 'none' else args.name
    save_folder = os.path.join(args.save_folder, exp_name)
    os.makedirs(save_folder, exist_ok=True)

    log_file = os.path.join(save_folder, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Using device: {device}")
    logging.info(f"Arguments: {vars(args)}")

    # Dataset & DataLoader
    dataset = utils.StateTransitionsDataset(hdf5_file=args.dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_cuda
    )
    logging.info(f"Loaded dataset")

    # Infer input image spatial dimensions (C, H, W)
    sample_obs = next(iter(train_loader))[0]
    input_shape = sample_obs.shape[1:]

    # Model Initialization
    model = models.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_slots=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        # encoder=args.encoder
    ).to(device)

    model.apply(utils.weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training Loop
    logging.info('Starting C-SWM model training...')
    best_loss = float('inf')
    model_file = os.path.join(save_folder, 'model.pt')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        for batch_idx, data_batch in enumerate(train_loader):
            # Move batch tensors to target device: (obs, action, next_obs)
            data_batch = [tensor.to(device) for tensor in data_batch]

            optimizer.zero_grad()

            # Forward pass: compute hinge contrastive loss
            loss = model.contrastive_loss(*data_batch)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_epoch_loss += batch_loss

            if batch_idx % args.log_interval == 0:
                percent_complete = 100.0 * batch_idx / len(train_loader)
                logging.info(
                    f"Epoch: {epoch:3d} [{batch_idx * len(data_batch[0]):5d}/{len(dataset):5d} "
                    f"({percent_complete:3.0f}%)]\tLoss: {batch_loss:.6f}"
                )

        avg_loss = total_epoch_loss / len(train_loader)
        logging.info(f"====> Epoch: {epoch:3d} | Average Loss: {avg_loss:.6f}")

        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)
            logging.info(f"--> Saved new best checkpoint (Loss: {best_loss:.6f}) to {model_file}")


if __name__ == '__main__':
    main()

# uv run train.py --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name pong
# uv run eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/pong --num-steps 1
