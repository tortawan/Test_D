"""
DQN Agent for Gymnasium's LunarLander-v3 Environment using PyTorch.
This version logs hyperparameters and evaluation results to a PostgreSQL database
using a two-table schema.
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

# --- Environment & Hyperparameter Setup from Environment Variables ---
DB_NAME = os.getenv("POSTGRES_DB", "experiments")
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "db")
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", f"exp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

# --- Core Hyperparameters ---
exp_spec: Dict[str, Any] = {
    'first_hid': int(os.getenv('FIRST_HID', 350)),
    'second_hid': int(os.getenv('SECOND_HID', 150)),
    'batch_size': int(os.getenv('BATCH_SIZE', 512)),
    'gamma': float(os.getenv('GAMMA', 0.99)),
    'learning_rate': float(os.getenv('LEARNING_RATE', 0.0005)),
    'epsilon_decay': float(os.getenv('EPSILON_DECAY', 0.99)),
    'replay_memory_capacity': int(os.getenv('REPLAY_MEMORY_CAPACITY', 100000)),
    'loss_function_type': 'mse',
    'target_network': False,
    'step_to_update': 2,
}

# --- Training Configuration ---
EPISODES: int = 500
TARGET_SCORE_AVG: float = 250
MAX_STEPS_PER_EPISODE: int = 1000
EVAL_EPISODES_COUNT: int = 20
EVALUATION_FREQUENCY: int = 10

# --- File Paths ---
OUTPUTS_DIR = "/app/outputs"
MODEL_SAVE_PATH: str = f"{OUTPUTS_DIR}/lunar_lander_dqn_{EXPERIMENT_ID}.pth"
PLOT_PATH: str = f"{OUTPUTS_DIR}/lunar_lander_plot_{EXPERIMENT_ID}.png"

# --- Basic Setup ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Database Functions ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database with retries."""
    for _ in range(5):
        try:
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432")
            logger.info("Successfully connected to PostgreSQL database.")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Could not connect to database: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    logger.error("Failed to connect to the database after several retries.")
    return None

def setup_database_tables(conn):
    """Creates the experiments and evaluation_results tables if they don't exist."""
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id SERIAL PRIMARY KEY,
                    experiment_id VARCHAR(255) UNIQUE NOT NULL,
                    first_hid INT,
                    second_hid INT,
                    batch_size INT,
                    gamma FLOAT,
                    learning_rate FLOAT,
                    epsilon_decay FLOAT,
                    replay_memory_capacity INT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id SERIAL PRIMARY KEY,
                    experiment_run_id INT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    episode INT NOT NULL,
                    avg_eval_score FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("Database tables are ready.")
    except psycopg2.Error as e:
        # This handles the race condition where both containers try to create the table.
        # If the table already exists, we just log it and continue.
        logger.warning(f"Could not create tables (they might already exist): {e}")
        conn.rollback()


def get_or_create_experiment(conn, config: Dict[str, Any]) -> Optional[int]:
    """Inserts a new experiment record if it doesn't exist and returns its ID."""
    if not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM experiments WHERE experiment_id = %s;", (EXPERIMENT_ID,))
            result = cur.fetchone()
            if result:
                logger.info(f"Experiment '{EXPERIMENT_ID}' already exists with ID {result[0]}.")
                return result[0]
            
            query = sql.SQL("""
                INSERT INTO experiments (experiment_id, first_hid, second_hid, batch_size, gamma, learning_rate, epsilon_decay, replay_memory_capacity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """)
            cur.execute(query, (
                EXPERIMENT_ID,
                config['first_hid'], config['second_hid'], config['batch_size'], config['gamma'],
                config['learning_rate'], config['epsilon_decay'], config['replay_memory_capacity']
            ))
            exp_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Created new experiment '{EXPERIMENT_ID}' with ID {exp_id}.")
            return exp_id
    except Exception as e:
        logger.error(f"Error creating experiment record: {e}")
        conn.rollback()
        return None

def log_evaluation_to_database(conn, experiment_run_id: int, episode: int, avg_score: float):
    """Logs an evaluation result to the database."""
    if not conn or experiment_run_id is None: return
    try:
        with conn.cursor() as cur:
            query = sql.SQL("""
                INSERT INTO evaluation_results (experiment_run_id, episode, avg_eval_score)
                VALUES (%s, %s, %s);
            """)
            cur.execute(query, (experiment_run_id, episode, avg_score))
            conn.commit()
    except Exception as e:
        logger.error(f"Error logging evaluation result to database: {e}")
        conn.rollback()

# --- QNetwork and DQNAgent Classes ---
class QNetwork(nn.Module):
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
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any], agent_device: torch.device):
        self.config = config
        self.state_size, self.action_size = state_size, action_size
        self.device = agent_device
        self.memory = deque(maxlen=self.config['replay_memory_capacity'])
        self.gamma = self.config['gamma']
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.01, self.config['epsilon_decay']
        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.model = QNetwork(state_size, action_size, self.config['first_hid'], self.config['second_hid']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    def remember(self, state, action, reward, next_state, done): self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
    def act(self, state_np, explore=True):
        if explore and random.random() <= self.epsilon: return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state_np.flatten()).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad(): action_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(action_values[0]).item()
    def replay(self):
        if len(self.memory) < self.batch_size: return 0.0
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Unpack and convert batch to tensors explicitly for robustness
        states_np = np.vstack([e[0] for e in minibatch])
        actions_np = np.array([e[1] for e in minibatch])
        rewards_np = np.array([e[2] for e in minibatch])
        next_states_np = np.vstack([e[3] for e in minibatch])
        dones_np = np.array([e[4] for e in minibatch])

        states = torch.from_numpy(states_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).long().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(rewards_np).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(next_states_np).float().to(self.device)
        dones = torch.from_numpy(dones_np.astype(np.uint8)).float().unsqueeze(-1).to(self.device)

        current_q = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0].unsqueeze(-1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()
    def save(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(self.model.state_dict(), filepath)
            logger.info(f"Saved model to {filepath}")
        except Exception as e: logger.error(f"Error saving model: {e}")

# --- Evaluation and Plotting ---
def run_evaluation_phase(env, agent, num_episodes, max_steps) -> float:
    scores = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        score = 0
        for _ in range(max_steps):
            action = agent.act(state, explore=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            score += reward
            if terminated or truncated: break
        scores.append(score)
    
    mean_score = np.mean(scores) if scores else -float('inf')
    # *** FIX: Convert numpy float to standard Python float ***
    return float(mean_score)

def plot_results(db_conn, experiment_run_id):
    if not db_conn or experiment_run_id is None: return
    df = pd.read_sql(f"SELECT episode, avg_eval_score FROM evaluation_results WHERE experiment_run_id = {experiment_run_id} ORDER BY episode", db_conn)
    if df.empty:
        logger.warning("No evaluation data to plot.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['avg_eval_score'], marker='o', linestyle='-', label='Avg Eval Score')
    plt.title(f'Evaluation Scores for {EXPERIMENT_ID}', fontsize=16)
    plt.xlabel('Training Episode', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.grid(True); plt.legend(); plt.tight_layout()
    try:
        os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
        plt.savefig(PLOT_PATH)
        logger.info(f"Plot saved to {PLOT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")

# --- Main Execution ---
def main():
    logger.info(f"Starting experiment: {EXPERIMENT_ID}")
    logger.info(f"Hyperparameters: {exp_spec}")
    
    db_conn = get_db_connection()
    setup_database_tables(db_conn)
    experiment_run_id = get_or_create_experiment(db_conn, exp_spec)

    if experiment_run_id is None:
        logger.error("Could not obtain a valid experiment ID. Aborting.")
        return

    env = gym.make("LunarLander-v3")
    agent = DQNAgentPyTorch(env.observation_space.shape[0], env.action_space.n, config=exp_spec, agent_device=device)

    for e in range(1, EPISODES + 1):
        state, _ = env.reset()
        score, loss_sum, replay_ops = 0, 0, 0
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if len(agent.memory) >= agent.batch_size and e % agent.config['step_to_update'] == 0:
                loss_sum += agent.replay(); replay_ops += 1
            if done: break
        
        avg_loss = (loss_sum / replay_ops) if replay_ops > 0 else 0
        logger.info(f"Exp: {EXPERIMENT_ID} | Ep: {e}/{EPISODES} | Score: {score:.2f} | Epsilon: {agent.epsilon:.3f} | Avg Loss: {avg_loss:.4f}")

        if e % EVALUATION_FREQUENCY == 0:
            avg_eval_score = run_evaluation_phase(env, agent, EVAL_EPISODES_COUNT, MAX_STEPS_PER_EPISODE)
            logger.info(f"--- Evaluation after Ep {e} | Avg Score: {avg_eval_score:.2f} ---")
            log_evaluation_to_database(db_conn, experiment_run_id, e, avg_eval_score)
            if avg_eval_score >= TARGET_SCORE_AVG:
                logger.info(f"Environment SOLVED after episode {e}!")
                break
    
    agent.save(MODEL_SAVE_PATH)
    plot_results(db_conn, experiment_run_id)
    env.close()
    if db_conn: db_conn.close()
    logger.info(f"Experiment {EXPERIMENT_ID} finished.")

if __name__ == "__main__":
    main()
