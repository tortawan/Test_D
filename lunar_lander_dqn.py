"""
DQN Agent for Gymnasium's LunarLander-v3 Environment using PyTorch.

This script implements a Deep Q-Network (DQN) agent to solve the
LunarLander-v3 environment. Key features include:
- A Q-Network model built with PyTorch.
- An agent implementing epsilon-greedy exploration, experience replay, and learning.
- Preliminary evaluation of a pre-trained model.
- A main training loop with periodic evaluation of the agent's performance.
- Early stopping if the target score is achieved during periodic evaluations.
- Saving and loading of model weights.
- Plotting of training progress (training scores and evaluation scores).
- Logging of training details and results to a CSV file.

The agent can be configured via the `exp_spec` dictionary and various constants
at the beginning of the script.
"""

import random
import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from typing import Tuple, List, Dict, Any, Optional, Deque

# Workaround for OpenMP runtime error: "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# This can occur in environments with multiple OpenMP libraries (e.g., if NumPy and PyTorch use different ones).
# Setting this environment variable allows multiple initializations, which can resolve the error on some systems.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Configuration Constants ---
EPISODES: int = 300  # Maximum number of training episodes
TARGET_SCORE_AVG: float = 250  # Target average score for considering the environment solved
MAX_STEPS_PER_EPISODE: int = 1000  # Max steps per episode (for training and evaluation)
EVAL_EPISODES_COUNT: int = 30  # Number of episodes for pre-training evaluation AND for periodic evaluation runs
EVALUATION_FREQUENCY: int = 5  # Evaluate the agent every N training episodes

# --- File Paths ---
MODEL_SAVE_PATH: str = "lunar_lander_dqn_pytorch_showcase.pth"
RESULTS_CSV_PATH: str = "lunar_lander_training_results_pytorch_showcase.csv"
PLOT_PATH: str = "lunar_lander_training_plot_pytorch_showcase.png"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Device Configuration ---
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- DQN Agent Hyperparameters ---
# For a production-grade DQN, consider using a separate target network (target_network: True)
# and potentially more advanced techniques like Double DQN or Dueling DQN.
exp_spec: Dict[str, Any] = {
    'first_hid': 350,                      # Neurons in the first hidden layer
    'second_hid': 150,                     # Neurons in the second hidden layer
    'loss_function_type': 'mse',           # Loss function (Mean Squared Error)
    'target_network': False,               # If True, a separate target network would be used (standard DQN).
                                           # Set to False for a simpler version where the main network estimates target Q-values.
    'batch_size': 512,                     # Number of experiences to sample from memory for each learning step
    'gamma': 0.99,                         # Discount factor for future rewards
    'learning_rate': 0.0005,               # Learning rate for the optimizer
    'step_to_update': 2,                   # Number of environment steps before a learning update
    'epsilon_decay': 0.99,                 # Multiplicative factor for decaying epsilon (exploration rate)
    'replay_memory_capacity': 100000        # Maximum number of experiences to store in replay memory
}


class QNetwork(nn.Module):
    """
    Neural Network for Q-value approximation.

    A simple feedforward neural network with two hidden layers and ReLU activations.
    """
    def __init__(self, state_size: int, action_size: int, first_hid: int, second_hid: int):
        """
        Initializes the QNetwork.

        Args:
            state_size (int): Dimension of the input state space.
            action_size (int): Dimension of the output action space (number of possible actions).
            first_hid (int): Number of neurons in the first hidden layer.
            second_hid (int): Number of neurons in the second hidden layer.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, first_hid)
        self.fc2 = nn.Linear(first_hid, second_hid)
        self.fc3 = nn.Linear(second_hid, action_size)
        logger.info(
            f"QNetwork initialized: Linear({state_size}, {first_hid}) -> ReLU -> "
            f"Linear({first_hid}, {second_hid}) -> ReLU -> Linear({second_hid}, {action_size})"
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgentPyTorch:
    """
    Deep Q-Network Agent implemented with PyTorch.

    This agent learns to interact with an environment using a Q-network,
    experience replay, and an epsilon-greedy exploration strategy.
    """
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any], agent_device: torch.device):
        """
        Initializes the DQN Agent.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            config (Dict[str, Any]): Dictionary containing hyperparameters for the agent.
            agent_device (torch.device): The device (CPU or CUDA) to run the agent on.
        """
        self.config: Dict[str, Any] = config
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.device: torch.device = agent_device

        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.config['replay_memory_capacity'])
        self.gamma: float = self.config['gamma']
        self.epsilon: float = 1.0  # Initial exploration rate
        self.epsilon_min: float = 0.01  # Minimum exploration rate
        self.epsilon_decay: float = self.config['epsilon_decay']
        self.learning_rate: float = self.config['learning_rate']
        self.batch_size: int = self.config['batch_size']

        # Initialize the Q-Network (policy network) and move it to the specified device
        self.model: QNetwork = QNetwork(
            state_size,
            action_size,
            self.config['first_hid'],
            self.config['second_hid']
        ).to(self.device)

        # Initialize the optimizer
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Set the loss function
        if self.config['loss_function_type'].lower() == 'mse':
            self.loss_fn: nn.Module = nn.MSELoss()
        else:
            logger.warning(f"Unsupported loss function type: {self.config['loss_function_type']}. Defaulting to MSE.")
            self.loss_fn = nn.MSELoss()

        logger.info("DQNAgentPyTorch initialized. Configuration:")
        for key, val in self.config.items():
            logger.info(f"  {key}: {val}")
        logger.info(f"  epsilon_initial: {self.epsilon}, epsilon_min: {self.epsilon_min}")
        logger.info(f"Model architecture:\n{self.model}")

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Stores an experience tuple in the replay memory.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observed.
            done (bool): Whether the episode has terminated.
        """
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

    def act(self, state_np: np.ndarray, explore: bool = True) -> int:
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state_np (np.ndarray): The current state as a NumPy array.
            explore (bool, optional): If True, allows for exploration (epsilon-greedy).
                                      If False, acts greedily (used for evaluation). Defaults to True.

        Returns:
            int: The action selected by the agent.
        """
        if explore and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        # Ensure state is 1D for processing
        if state_np.ndim > 1:
            state_np = state_np.flatten()

        # Convert state to a PyTorch tensor, add a batch dimension, and move to device
        state_tensor: torch.Tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)

        # Get Q-values from the model (exploitation)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations for inference
            action_values: torch.Tensor = self.model(state_tensor)
        self.model.train()  # Set the model back to training mode

        return torch.argmax(action_values[0]).item()

    def replay(self) -> float:
        """
        Performs a learning update using a batch of experiences from the replay memory.

        Returns:
            float: The loss value for the current training step. Returns 0.0 if not enough samples in memory.
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Randomly sample a minibatch of experiences
        minibatch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(self.memory, self.batch_size)

        # Unpack the minibatch and convert to PyTorch tensors
        states_np = np.array([experience[0] for experience in minibatch])
        actions_np = np.array([experience[1] for experience in minibatch])
        rewards_np = np.array([experience[2] for experience in minibatch])
        next_states_np = np.array([experience[3] for experience in minibatch])
        dones_np = np.array([experience[4] for experience in minibatch])

        states_tensor = torch.from_numpy(states_np).float().to(self.device)
        actions_tensor = torch.from_numpy(actions_np).long().unsqueeze(-1).to(self.device)
        rewards_tensor = torch.from_numpy(rewards_np).float().unsqueeze(-1).to(self.device)
        next_states_tensor = torch.from_numpy(next_states_np).float().to(self.device)
        dones_tensor = torch.from_numpy(dones_np.astype(np.uint8)).float().unsqueeze(-1).to(self.device)

        # Get current Q-values for the actions taken
        current_q_values: torch.Tensor = self.model(states_tensor).gather(1, actions_tensor)

        # Calculate target Q-values
        with torch.no_grad():
            # If self.config['target_network'] were True, a separate target model would be used here.
            # Currently, the main model is used for simplicity.
            next_q_values_all_actions: torch.Tensor = self.model(next_states_tensor)
            max_next_q_values: torch.Tensor = next_q_values_all_actions.max(1)[0].unsqueeze(-1)
            target_q_values: torch.Tensor = rewards_tensor + (self.gamma * max_next_q_values * (1 - dones_tensor))

        # Calculate loss
        loss: torch.Tensor = self.loss_fn(current_q_values, target_q_values)

        # Perform backpropagation and optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        
        return loss.item()

    def load(self, filepath: str) -> bool:
        """
        Loads model weights from a file.

        Args:
            filepath (str): The path to the file containing model weights.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.eval()  # Set model to evaluation mode after loading
            logger.info(f"Successfully loaded model weights from {filepath}")
            return True
        except FileNotFoundError:
            logger.warning(f"Weight file not found at {filepath}. Starting with a new model.")
            return False
        except Exception as e:
            logger.error(f"Error loading model weights from {filepath}: {e}")
            return False

    def save(self, filepath: str) -> None:
        """
        Saves model weights to a file.

        Args:
            filepath (str): The path where model weights will be saved.
        """
        try:
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Successfully saved model weights to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model weights to {filepath}: {e}")


def plot_training_results(df_results: pd.DataFrame, plot_path: str, title_suffix: str = "") -> None:
    """
    Plots training progress and saves the plot to a file.

    The plot includes:
    - Score per training episode.
    - Average score from periodic evaluation runs (if available).
    - Cumulative average score from pre-training evaluation (if training was skipped).

    Args:
        df_results (pd.DataFrame): DataFrame containing training results.
                                   Expected columns: 'episode', 'training_reward',
                                   and a column for average evaluation scores
                                   (e.g., 'average_post_training_eval_score' or 'cumulative_average_eval_score').
        plot_path (str): Path to save the generated plot.
        title_suffix (str, optional): Suffix to append to the plot title. Defaults to "".
    """
    plt.figure(figsize=(12, 6)) 

    avg_reward_col_to_plot: Optional[str] = None
    avg_reward_legend_label: str = 'Average Score Trend' 

    # Determine which average reward column to plot based on availability and priority
    if "cumulative_average_eval_score" in df_results.columns:
        avg_reward_col_to_plot = "cumulative_average_eval_score"
        avg_reward_legend_label = 'Cumulative Avg Eval Score (Pre-Training)'
    elif "average_post_training_eval_score" in df_results.columns:
        avg_reward_col_to_plot = "average_post_training_eval_score"
        avg_reward_legend_label = f'Avg Score ({EVAL_EPISODES_COUNT} Eval Eps, every {EVALUATION_FREQUENCY} Train Eps)'
    
    # Plot score per TRAINING episode
    individual_score_col: str = 'training_reward'
    if individual_score_col not in df_results.columns and 'reward' in df_results.columns:
        individual_score_col = 'reward' # Fallback for pre-evaluation data

    if individual_score_col in df_results.columns and not df_results[individual_score_col].empty:
        plt.plot(df_results['episode'], df_results[individual_score_col], label='Training Score per Episode', alpha=0.6, color='deepskyblue', linestyle='-')

    # Plot the chosen average reward trend (NaNs will create gaps)
    if avg_reward_col_to_plot and avg_reward_col_to_plot in df_results.columns and \
       not df_results[avg_reward_col_to_plot].dropna().empty:
        plt.plot(df_results['episode'], df_results[avg_reward_col_to_plot], label=avg_reward_legend_label, color='darkorange', linewidth=2, marker='o', markersize=5, linestyle='-')

    plt.title(f'Lunar Lander DQN Training Progress (PyTorch){title_suffix}', fontsize=16)
    plt.xlabel('Training Episode Number', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
    try:
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {plot_path}: {e}")
    # plt.show() # Uncomment to display the plot directly in interactive environments


def run_evaluation_phase(
    env: gym.Env, 
    agent: DQNAgentPyTorch, 
    num_episodes: int, 
    max_steps: int,
    phase_name: str = "Evaluation"
) -> Tuple[List[float], float]:
    """
    Runs an evaluation phase for the agent.

    Args:
        env (gym.Env): The Gymnasium environment.
        agent (DQNAgentPyTorch): The DQN agent to evaluate.
        num_episodes (int): The number of episodes to run for evaluation.
        max_steps (int): The maximum number of steps per evaluation episode.
        phase_name (str, optional): Name of the evaluation phase for logging. Defaults to "Evaluation".

    Returns:
        Tuple[List[float], float]: A list of scores from each evaluation episode,
                                   and the average score over all evaluation episodes.
    """
    logger.info(f"--- Starting {phase_name} Phase ({num_episodes} episodes) ---")
    eval_rewards_list: List[float] = []
    for eval_ep in range(1, num_episodes + 1):
        # Gymnasium's env.reset() returns a tuple (observation, info)
        current_state_tuple: Tuple[np.ndarray, Dict] = env.reset()
        current_state_np: np.ndarray = current_state_tuple[0]
        
        eval_episode_reward: float = 0.0
        for _step_eval in range(max_steps):
            action: int = agent.act(current_state_np, explore=False) # Act greedily
            
            # Gymnasium's env.step() returns (observation, reward, terminated, truncated, info)
            next_state_tuple, reward, terminated, truncated, _info = env.step(action)
            done: bool = terminated or truncated
            
            current_state_np = next_state_tuple # In new Gymnasium, next_state_tuple is already the observation array
            eval_episode_reward += reward
            if done:
                break
        eval_rewards_list.append(eval_episode_reward)
        logger.info(f"{phase_name} Episode: {eval_ep}/{num_episodes} | Score: {eval_episode_reward:.2f}")
    
    avg_eval_reward: float = np.mean(eval_rewards_list).item() if eval_rewards_list else -float('inf')
    logger.info(f"--- {phase_name} Phase Complete | Average Score: {avg_eval_reward:.2f} over {len(eval_rewards_list)} episodes ---")
    return eval_rewards_list, avg_eval_reward


def main() -> None:
    """
    Main function to run the DQN agent training and evaluation.
    """
    script_start_time = datetime.datetime.now()
    logger.info(f"Starting PyTorch DQN script for LunarLander-v3 at {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Environment Setup ---
    try:
        env: gym.Env = gym.make("LunarLander-v3")
        logger.info("Gymnasium environment 'LunarLander-v3' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create Gymnasium environment 'LunarLander-v3': {e}")
        logger.info("Please ensure 'gymnasium[box2d]' is installed (e.g., pip install 'gymnasium[box2d]').")
        return # Exit if environment creation fails

    state_size: int = env.observation_space.shape[0]
    action_size: int = env.action_space.n

    # --- Agent Initialization ---
    agent: DQNAgentPyTorch = DQNAgentPyTorch(state_size, action_size, config=exp_spec, agent_device=device)

    skip_training: bool = False
    pre_evaluation_scores_for_df: List[float] = []

    # --- Preliminary Evaluation Phase (if model exists) ---
    logger.info(f"Attempting to load model from {MODEL_SAVE_PATH} for Preliminary Evaluation.")
    weights_loaded_successfully: bool = agent.load(MODEL_SAVE_PATH)

    if weights_loaded_successfully:
        pre_eval_scores, avg_pre_eval_score = run_evaluation_phase(
            env, agent, EVAL_EPISODES_COUNT, MAX_STEPS_PER_EPISODE, "Preliminary Evaluation"
        )
        pre_evaluation_scores_for_df = pre_eval_scores
        if avg_pre_eval_score >= TARGET_SCORE_AVG:
            logger.info(
                f"Target score {TARGET_SCORE_AVG:.2f} met or exceeded during preliminary evaluation "
                f"(Avg: {avg_pre_eval_score:.2f}). Skipping training."
            )
            skip_training = True
        else:
            logger.info(
                f"Preliminary evaluation average score ({avg_pre_eval_score:.2f}) is below target. "
                "Proceeding with training."
            )
            agent.epsilon = 1.0 # Reset epsilon for new training session
    else:
        logger.info("No existing model weights loaded or error during load. Proceeding directly to training.")

    # --- Main Training Loop ---
    all_training_scores: List[float] = []
    all_avg_losses: List[float] = []
    all_post_training_eval_avg_scores: List[Optional[float]] = [] # Stores eval scores or NaN

    if not skip_training:
        logger.info(f"--- Starting Main Training Phase: {EPISODES} episodes, Evaluate every {EVALUATION_FREQUENCY} episodes ---")
        total_steps_counter: int = 0
        solved_early: bool = False
        
        # For more accurate epsilon logging, one could log agent.epsilon at the end of each episode.
        # The current `report_epsilons` in results handling is an approximation.

        for e in range(1, EPISODES + 1): # Loop over training episodes
            # --- Training Episode ---
            current_state_tuple: Tuple[np.ndarray, Dict] = env.reset()
            current_state_np: np.ndarray = current_state_tuple[0]
            
            training_episode_reward: float = 0.0
            episode_loss_sum: float = 0.0
            num_replay_ops_this_episode: int = 0
            steps_this_training_episode: int = 0

            for step_in_episode in range(MAX_STEPS_PER_EPISODE):
                action: int = agent.act(current_state_np, explore=True)
                next_state_tuple, reward, terminated, truncated, _info = env.step(action)
                done: bool = terminated or truncated
                
                next_state_np_processed: np.ndarray = next_state_tuple

                agent.remember(current_state_np, action, reward, next_state_np_processed, done)
                current_state_np = next_state_np_processed
                training_episode_reward += reward
                total_steps_counter += 1
                steps_this_training_episode = step_in_episode + 1

                if len(agent.memory) >= agent.batch_size and \
                   total_steps_counter % agent.config['step_to_update'] == 0:
                    loss_from_replay: float = agent.replay()
                    if loss_from_replay is not None: # replay() returns 0.0 if not enough memory
                        episode_loss_sum += loss_from_replay
                        num_replay_ops_this_episode += 1
                
                if done:
                    break
            
            avg_episode_loss: float = (episode_loss_sum / num_replay_ops_this_episode) if num_replay_ops_this_episode > 0 else 0.0
            all_training_scores.append(training_episode_reward)
            all_avg_losses.append(avg_episode_loss)
            
            logger.info(
                f"Training Episode: {e}/{EPISODES} | Training Score: {training_episode_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | Avg Loss: {avg_episode_loss:.4f} | "
                f"Steps: {steps_this_training_episode}"
            )

            # --- Periodic Evaluation Run ---
            avg_score_this_eval_run: Optional[float] = np.nan # Default if no eval this episode

            if e % EVALUATION_FREQUENCY == 0:
                _eval_scores, avg_score_this_eval_run = run_evaluation_phase(
                    env, agent, EVAL_EPISODES_COUNT, MAX_STEPS_PER_EPISODE, f"Periodic Eval (after Ep {e})"
                )
                if avg_score_this_eval_run is not None and avg_score_this_eval_run >= TARGET_SCORE_AVG:
                    logger.info(
                        f"\nEnvironment SOLVED after training episode {e}! "
                        f"Average score over {EVAL_EPISODES_COUNT} evaluation episodes: {avg_score_this_eval_run:.2f} >= {TARGET_SCORE_AVG:.2f}."
                    )
                    agent.save(MODEL_SAVE_PATH)
                    solved_early = True
            
            all_post_training_eval_avg_scores.append(avg_score_this_eval_run)

            # Save model checkpoint periodically (e.g., every 50 episodes)
            if e % 50 == 0:
                checkpoint_path = MODEL_SAVE_PATH.replace(".pth", f"_ep{e}.pth")
                agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint model to {checkpoint_path}")
            
            if solved_early:
                break # Exit main training loop
        
        # Save final model if not solved early or if loop completed
        if not solved_early and not skip_training:
            logger.info(f"Training completed or target score not reached. Saving final model at episode {e}.")
            agent.save(MODEL_SAVE_PATH)

    env.close()
    logger.info("Environment closed.")
    script_end_time = datetime.datetime.now()
    total_script_time = script_end_time - script_start_time
    logger.info(f"Script finished at {script_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total script execution time: {str(total_script_time)}")

    # --- Results Handling and Saving ---
    logger.info("--- Preparing Results for Reporting ---")
    plot_title_suffix: str = ""
    df_avg_reward_col_name: str = 'average_post_training_eval_score'
    avg_reward_series_for_df: List[Optional[float]] = []
    
    report_scores_list: List[float] = []
    report_avg_losses_list: List[float] = []
    report_epsilons_list: List[float] = [] # Stores approximate epsilon values

    if skip_training and weights_loaded_successfully:
        report_scores_list = pre_evaluation_scores_for_df
        report_avg_losses_list = [0.0] * len(pre_evaluation_scores_for_df)
        report_epsilons_list = [0.0] * len(pre_evaluation_scores_for_df)
        plot_title_suffix = " (Pre-Evaluation Results)"
        logger.info("Reporting pre-evaluation results as training was skipped.")
        if report_scores_list:
            avg_reward_series_for_df = pd.Series(report_scores_list).expanding().mean().tolist()
            df_avg_reward_col_name = 'cumulative_average_eval_score'
        else:
            avg_reward_series_for_df = []

    elif not all_training_scores:
        logger.info("No training scores to report (training might not have run).")
    else: # Training occurred
        report_scores_list = all_training_scores
        report_avg_losses_list = all_avg_losses
        plot_title_suffix = f" (Training, Eval every {EVALUATION_FREQUENCY} eps)"
        avg_reward_series_for_df = all_post_training_eval_avg_scores # Contains NaNs for non-eval episodes
        
        # Approximate epsilon values for reporting (actual epsilon changes during replay)
        temp_epsilon: float = 1.0
        min_eps: float = agent.epsilon_min
        decay: float = agent.config['epsilon_decay']
        step_to_update_freq: int = agent.config['step_to_update']
        
        for i in range(len(report_scores_list)):
            report_epsilons_list.append(temp_epsilon)
            # This is a rough approximation of epsilon decay per episode.
            # Actual decay depends on the number of replay calls.
            if len(agent.memory) >= agent.batch_size: # Check if replay could have happened
                # Estimate number of replay calls in an episode (highly approximate)
                num_replays_in_episode_approx: int = MAX_STEPS_PER_EPISODE // step_to_update_freq
                for _ in range(num_replays_in_episode_approx):
                    if temp_epsilon > min_eps:
                        temp_epsilon *= decay
                        temp_epsilon = max(min_eps, temp_epsilon)
        logger.info(f"Reporting training run results (with evaluations every {EVALUATION_FREQUENCY} episodes).")

    if report_scores_list:
        num_reported_episodes = len(report_scores_list)
        data_for_df: Dict[str, List[Any]] = {
            'episode': list(range(1, num_reported_episodes + 1)),
            'training_reward': report_scores_list,
            'average_loss': report_avg_losses_list,
            'epsilon_approx': report_epsilons_list[:num_reported_episodes] # Ensure length match
        }
        
        # Ensure avg_reward_series_for_df has the correct length
        if len(avg_reward_series_for_df) == num_reported_episodes:
            data_for_df[df_avg_reward_col_name] = avg_reward_series_for_df
        elif skip_training and avg_reward_series_for_df: # Special case for pre-eval
             data_for_df[df_avg_reward_col_name] = avg_reward_series_for_df
        else: # Fallback if lengths mismatch
            logger.warning(
                f"Length mismatch for average reward series. Scores: {num_reported_episodes}, "
                f"Avg Series: {len(avg_reward_series_for_df)}. Filling '{df_avg_reward_col_name}' with NaNs."
            )
            data_for_df[df_avg_reward_col_name] = [np.nan] * num_reported_episodes

        results_df = pd.DataFrame(data_for_df)
        try:
            results_df.to_csv(RESULTS_CSV_PATH, index=False)
            logger.info(f"Results saved to {RESULTS_CSV_PATH}")
        except Exception as e:
            logger.error(f"Failed to save results CSV to {RESULTS_CSV_PATH}: {e}")

        plot_training_results(results_df, PLOT_PATH, title_suffix=plot_title_suffix)
    else:
        logger.info("No scores available to generate CSV or plot.")

    logger.info("Script execution fully finished.")


if __name__ == "__main__":
    main()
