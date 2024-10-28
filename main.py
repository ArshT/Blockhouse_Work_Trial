from code.train import train_func
import argparse


def main():
    # Get all the arguments 
    parser = argparse.ArgumentParser(description='Train a DQN model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Replay buffer size')
    parser.add_argument('--learning_starts', type=int, default=1000, help='Number of steps before learning starts')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--tau', type=float, default=1.0, help='Tau')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')
    parser.add_argument('--train_freq', type=int, default=4, help='Train frequency')
    parser.add_argument('--target_update_interval', type=int, default=1000, help='Target update interval')
    parser.add_argument('--exploration_fraction', type=float, default=0.1, help='Exploration fraction')
    parser.add_argument('--exploration_final_eps', type=float, default=0.01, help='Exploration final epsilon')
    parser.add_argument('--save_freq', type=int, default=10000, help='Save frequency')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total timesteps')
    args = parser.parse_args()


    # Call the train function
    train_func(args)


if __name__ == '__main__':
    main()


