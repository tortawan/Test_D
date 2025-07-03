"""
DQN Agent for Gymnasium's LunarLander-v3 Environment using PyTorch.
This version is modified to log results to a PostgreSQL database.
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
import time
import psycopg2
from psycopg2 import sql
from typing import Tuple, List, Dict, Any, Optional, Deque

# --- Environment Setup for Docker ---
# These will be passed from docker-compose.yml
DB_NAME = os.getenv("POSTGRES_DB", "experiments")
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "db") # The service name of the database in docker-compose
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", f"exp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")


# Workaround for OpenMP runtime error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Configuration Constants ---
EPISODES: int = 500
TARGET_SCORE_AVG: float = 250
MAX_STEPS_PER_EPISODE: int = 1000
EVAL_EPISODES_COUNT: int = 20
EVALUATION_FREQUENCY: int = 5

# --- File Paths (for model and plot, CSV is replaced by DB) ---
MODEL_SAVE_PATH: str = f"/app/outputs/lunar_lander_dqn_{EXPERIMENT_ID}.pth"
PLOT_PATH: str = f"/app/outputs/lunar_lander_plot_{EXPERIMENT_ID}.png"

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
exp_spec: Dict[str, Any] = {
    'first_hid': 350,
    'second_hid': 150,
    'loss_function_type': 'mse',
    'target_network': False,
    'batch_size': 512,
    'gamma': 0.99,
    'learning_rate': 0.0005,
    'step_to_update': 2,
    'epsilon_decay': 0.99,
    'replay_memory_capacity': 100000
}

# --- Database Functions ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = None
    # Retry connection to give the DB time to start up
    for _ in range(5):
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port="5432"
            )
            logger.info("Successfully connected to PostgreSQL database.")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Could not connect to database: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    logger.error("Failed to connect to the database after several retries.")
    return None

def setup_database_table(conn):
    """Creates the results table if it doesn't exist."""
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    experiment_id VARCHAR(255) NOT NULL,
                    episode INT NOT NULL,
                    training_reward FLOAT,
                    average_loss FLOAT,
                    epsilon_approx FLOAT,
                    avg_eval_score FLOAT
                );
            """)
            conn.commit()
            logger.info("Database table 'experiment_results' is ready.")
    except Exception as e:
        logger.error(f"Error setting up database table: {e}")
        conn.rollback()


def log_to_database(conn, data: Dict[str, Any]):
    """Logs a dictionary of data to the database."""
    if not conn:
        return
    
    # Ensure required fields are present
    data.setdefault('experiment_id', EXPERIMENT_ID)
    data.setdefault('training_reward', None)
    data.setdefault('average_loss', None)
    data.setdefault('epsilon_approx', None)
    data.setdefault('avg_eval_score', None)

    try:
        with conn.cursor() as cur:
            query = sql.SQL("""
                INSERT INTO experiment_results (experiment_id, episode, training_reward, average_loss, epsilon_approx, avg_eval_score)
                VALUES (%s, %s, %s, %s, %s, %s);
            """)
            cur.execute(query, (
                data['experiment_id'],
                data['episode'],
                data['training_reward'],
                data['average_loss'],
                data['epsilon_approx'],
                data['avg_eval_score']
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error logging to database: {e}")
        conn.rollback()


class QNetwork(nn.Module):
    """Neural Network for Q-value approximation."""
    def __init__(self, state_size: int, action_size: int, first_hid: int, second_hid: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, first_hid)
        self.fc2 = nn.Linear(first_hid, second_hid)
        self.fc3 = nn.Linear(second_hid, action_size)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgentPyTorch:
    """Deep Q-Network Agent implemented with PyTorch."""
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any], agent_device: torch.device):
        self.config: Dict[str, Any] = config
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.device: torch.device = agent_device
        self.memory: Deque = deque(maxlen=self.config['replay_memory_capacity'])
        self.gamma: float = self.config['gamma']
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = self.config['epsilon_decay']
        self.learning_rate: float = self.config['learning_rate']
        self.batch_size: int = self.config['batch_size']
        self.model: QNetwork = QNetwork(state_size, action_size, self.config['first_hid'], self.config['second_hid']).to(self.device)
        self.optimizer: optim.Optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.config['loss_function_type'].lower() == 'mse':
            self.loss_fn: nn.Module = nn.MSELoss()
        else:
            self.loss_fn = nn.MSELoss()
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
    def act(self, state_np: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        if state_np.ndim > 1:
            state_np = state_np.flatten()
        state_tensor: torch.Tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values: torch.Tensor = self.model(state_tensor)
        self.model.train()
        return torch.argmax(action_values[0]).item()
    def replay(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        minibatch: List = random.sample(self.memory, self.batch_size)
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
        current_q_values: torch.Tensor = self.model(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values_all_actions: torch.Tensor = self.model(next_states_tensor)
            max_next_q_values: torch.Tensor = next_q_values_all_actions.max(1)[0].unsqueeze(-1)
            target_q_values: torch.Tensor = rewards_tensor + (self.gamma * max_next_q_values * (1 - dones_tensor))
        loss: torch.Tensor = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        return loss.item()
    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
             logger.warning(f"Weight file not found at {filepath}. Starting with a new model.")
             return False
        try:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.eval()
            logger.info(f"Successfully loaded model weights from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model weights from {filepath}: {e}")
            return False
    def save(self, filepath: str) -> None:
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Successfully saved model weights to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model weights to {filepath}: {e}")

def plot_training_results(df_results: pd.DataFrame, plot_path: str, title_suffix: str = "") -> None:
    plt.figure(figsize=(12, 6))
    if "training_reward" in df_results.columns:
        plt.plot(df_results['episode'], df_results['training_reward'], label='Training Score per Episode', alpha=0.6)
    if "avg_eval_score" in df_results.columns:
        plt.plot(df_results['episode'], df_results['avg_eval_score'].ffill(), label='Avg Eval Score', color='darkorange', linewidth=2)
    plt.title(f'Lunar Lander DQN Training Progress ({EXPERIMENT_ID}){title_suffix}', fontsize=16)
    plt.xlabel('Training Episode Number', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {plot_path}: {e}")

def run_evaluation_phase(env: gym.Env, agent: DQNAgentPyTorch, num_episodes: int, max_steps: int, phase_name: str = "Evaluation") -> Tuple[List[float], float]:
    logger.info(f"--- Starting {phase_name} Phase ({num_episodes} episodes) ---")
    eval_rewards_list: List[float] = []
    for eval_ep in range(1, num_episodes + 1):
        current_state_tuple: Tuple[np.ndarray, Dict] = env.reset()
        current_state_np: np.ndarray = current_state_tuple[0]
        eval_episode_reward: float = 0.0
        for _ in range(max_steps):
            action: int = agent.act(current_state_np, explore=False)
            next_state_tuple, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            current_state_np = next_state_tuple
            eval_episode_reward += reward
            if done:
                break
        eval_rewards_list.append(eval_episode_reward)
    avg_eval_reward: float = np.mean(eval_rewards_list).item() if eval_rewards_list else -float('inf')
    logger.info(f"--- {phase_name} Complete | Average Score: {avg_eval_reward:.2f} ---")
    return eval_rewards_list, avg_eval_reward

def main() -> None:
    """Main function to run the DQN agent training and evaluation."""
    logger.info(f"Starting PyTorch DQN script for LunarLander-v3. Experiment ID: {EXPERIMENT_ID}")

    # --- Database Setup ---
    db_conn = get_db_connection()
    setup_database_table(db_conn)

    # --- Environment Setup ---
    try:
        env: gym.Env = gym.make("LunarLander-v3")
    except Exception as e:
        logger.error(f"Failed to create Gymnasium environment: {e}")
        return

    state_size: int = env.observation_space.shape[0]
    action_size: int = env.action_space.n

    # --- Agent Initialization ---
    agent: DQNAgentPyTorch = DQNAgentPyTorch(state_size, action_size, config=exp_spec, agent_device=device)

    # --- Main Training Loop ---
    logger.info(f"--- Starting Main Training Phase: {EPISODES} episodes ---")
    
    all_results = [] # To store data for the final plot

    for e in range(1, EPISODES + 1):
        current_state_tuple: Tuple[np.ndarray, Dict] = env.reset()
        current_state_np: np.ndarray = current_state_tuple[0]
        training_episode_reward: float = 0.0
        episode_loss_sum: float = 0.0
        num_replay_ops: int = 0
        total_steps_counter = 0

        for step_in_episode in range(MAX_STEPS_PER_EPISODE):
            action: int = agent.act(current_state_np, explore=True)
            next_state_tuple, reward, terminated, truncated, _ = env.step(action)
            done: bool = terminated or truncated
            agent.remember(current_state_np, action, reward, next_state_tuple, done)
            current_state_np = next_state_tuple
            training_episode_reward += reward
            total_steps_counter += 1

            if len(agent.memory) >= agent.batch_size and total_steps_counter % agent.config['step_to_update'] == 0:
                loss = agent.replay()
                if loss is not None:
                    episode_loss_sum += loss
                    num_replay_ops += 1
            if done:
                break
        
        avg_episode_loss = (episode_loss_sum / num_replay_ops) if num_replay_ops > 0 else 0.0
        
        log_data = {
            "episode": e,
            "training_reward": training_episode_reward,
            "average_loss": avg_episode_loss,
            "epsilon_approx": agent.epsilon,
            "avg_eval_score": None # Placeholder
        }

        # --- Periodic Evaluation ---
        if e % EVALUATION_FREQUENCY == 0:
            _, avg_score_this_eval_run = run_evaluation_phase(
                env, agent, EVAL_EPISODES_COUNT, MAX_STEPS_PER_EPISODE, f"Periodic Eval (after Ep {e})"
            )
            log_data["avg_eval_score"] = avg_score_this_eval_run
            if avg_score_this_eval_run >= TARGET_SCORE_AVG:
                logger.info(f"Environment SOLVED after episode {e}!")
                agent.save(MODEL_SAVE_PATH)
                log_to_database(db_conn, log_data)
                all_results.append(log_data)
                break # End training
        
        logger.info(
            f"Exp: {EXPERIMENT_ID} | Episode: {e}/{EPISODES} | Score: {training_episode_reward:.2f} | "
            f"Epsilon: {agent.epsilon:.3f} | Avg Loss: {avg_episode_loss:.4f}"
        )
        
        # Log to DB
        log_to_database(db_conn, log_data)
        all_results.append(log_data)

        if e % 50 == 0:
            agent.save(MODEL_SAVE_PATH)

    # --- Finalization ---
    agent.save(MODEL_SAVE_PATH)
    env.close()
    if db_conn:
        db_conn.close()
    
    # Generate and save plot from collected results
    if all_results:
        results_df = pd.DataFrame(all_results)
        plot_training_results(results_df, PLOT_PATH)

    logger.info(f"Experiment {EXPERIMENT_ID} finished.")

if __name__ == "__main__":
    main()
