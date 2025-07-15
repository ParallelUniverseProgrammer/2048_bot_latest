"""
Mock data generator for realistic training updates and game states
"""
import random
import time
import math
from typing import Dict, List, Any
from datetime import datetime

class MockTrainingData:
    def __init__(self):
        self.episode_count = 0
        self.base_score = 1000
        self.loss_history = []
        self.score_history = []
        self.learning_rate = 0.0003
        self.current_board = self.generate_random_board()
        self.expert_usage = [0.12, 0.08, 0.15, 0.18, 0.10, 0.14, 0.11, 0.12]
        self.attention_weights = self.generate_attention_weights()
        
    def generate_random_board(self) -> List[List[int]]:
        """Generate a realistic 2048 board state"""
        board = [[0 for _ in range(4)] for _ in range(4)]
        
        # Add some tiles based on game progression
        num_tiles = random.randint(8, 14)
        positions = [(i, j) for i in range(4) for j in range(4)]
        random.shuffle(positions)
        
        for i in range(num_tiles):
            row, col = positions[i]
            # Weight towards smaller values with some larger ones
            if i < 4:
                board[row][col] = random.choice([2, 4, 8, 16])
            elif i < 8:
                board[row][col] = random.choice([16, 32, 64, 128])
            else:
                board[row][col] = random.choice([64, 128, 256, 512])
        
        # Occasionally add a high-value tile
        if random.random() < 0.3:
            empty_positions = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]
            if empty_positions:
                row, col = random.choice(empty_positions)
                board[row][col] = random.choice([512, 1024, 2048])
        
        return board
    
    def generate_attention_weights(self) -> List[List[float]]:
        """Generate realistic attention weights for 4x4 board"""
        weights = []
        for i in range(4):
            row = []
            for j in range(4):
                # Higher attention on corners and edges
                if (i == 0 or i == 3) and (j == 0 or j == 3):
                    weight = random.uniform(0.15, 0.25)
                elif i == 0 or i == 3 or j == 0 or j == 3:
                    weight = random.uniform(0.08, 0.15)
                else:
                    weight = random.uniform(0.02, 0.08)
                row.append(weight)
            weights.append(row)
        
        # Normalize weights
        total = sum(sum(row) for row in weights)
        for i in range(4):
            for j in range(4):
                weights[i][j] /= total
        
        return weights
    
    def update_expert_usage(self):
        """Update expert usage with some randomness"""
        for i in range(len(self.expert_usage)):
            # Small random changes
            change = random.uniform(-0.02, 0.02)
            self.expert_usage[i] = max(0.02, min(0.25, self.expert_usage[i] + change))
        
        # Normalize
        total = sum(self.expert_usage)
        self.expert_usage = [x / total for x in self.expert_usage]
    
    def generate_training_update(self, episode: int) -> Dict[str, Any]:
        """Generate a realistic training update"""
        self.episode_count = episode
        
        # Generate loss values with realistic trends
        if episode < 100:
            # Initial high loss with rapid decrease
            base_loss = 2.0 - (episode / 100) * 1.5
        elif episode < 1000:
            # Gradual decrease
            base_loss = 0.5 - (episode / 1000) * 0.3
        else:
            # Stabilized loss with small fluctuations
            base_loss = 0.2 + 0.1 * math.sin(episode / 100)
        
        # Add noise
        policy_loss = max(0.001, base_loss * 0.6 + random.uniform(-0.05, 0.05))
        value_loss = max(0.001, base_loss * 0.4 + random.uniform(-0.03, 0.03))
        total_loss = policy_loss + value_loss
        
        # Generate score with improvement trend
        base_score = self.base_score + (episode / 10) * 50
        score_noise = random.uniform(-200, 400)
        current_score = max(100, int(base_score + score_noise))
        
        # Generate rewards
        reward = current_score / 1000.0 + random.uniform(-0.5, 0.5)
        
        # Update histories
        self.loss_history.append(total_loss)
        self.score_history.append(current_score)
        
        # Keep only last 100 points for performance
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-100:]
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-100:]
        
        # Update board occasionally
        if episode % 5 == 0:
            self.current_board = self.generate_random_board()
        
        # Update expert usage
        self.update_expert_usage()
        
        # Update attention weights
        self.attention_weights = self.generate_attention_weights()
        
        # Generate action probabilities
        actions = [random.uniform(0.1, 0.4) for _ in range(4)]
        actions_sum = sum(actions)
        actions = [x / actions_sum for x in actions]
        
        # Calculate entropy
        entropy = -sum(p * math.log(p) for p in actions if p > 0)
        
        # GPU memory usage (simulate)
        gpu_memory = 4.2 + random.uniform(-0.3, 0.8)
        
        # Mock additional metrics
        avg_game_length = 45 + random.uniform(-10, 20)
        min_game_length = max(10, int(avg_game_length - random.uniform(10, 25)))
        max_game_length = int(avg_game_length + random.uniform(15, 40))
        wall_clock_elapsed = episode * 0.6 + random.uniform(-30, 30)  # Simulate ~0.6 seconds per episode
        
        # Estimate time to next checkpoint (every 100 episodes)
        episodes_to_checkpoint = 100 - (episode % 100)
        estimated_time_to_checkpoint = episodes_to_checkpoint * 0.6  # seconds
        
        return {
            "type": "training_update",
            "timestamp": time.time(),
            "episode": episode,
            "score": current_score,
            "reward": reward,
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "learning_rate": self.learning_rate,
            "actions": actions,
            "board_state": self.current_board,
            "attention_weights": self.attention_weights,
            "expert_usage": self.expert_usage,
            "gpu_memory": gpu_memory,
            "model_params": 45.6,  # Million parameters
            "loss_history": {
                "episodes": list(range(max(1, episode - len(self.loss_history) + 1), episode + 1))[-50:],
                "values": self.loss_history[-50:]
            },
            "score_history": {
                "episodes": list(range(max(1, episode - len(self.score_history) + 1), episode + 1))[-50:],
                "values": self.score_history[-50:]
            },
            "training_speed": random.uniform(95, 105),  # Episodes per minute
            "avg_game_length": avg_game_length,
            "min_game_length": min_game_length,
            "max_game_length": max_game_length,
            "wall_clock_elapsed": wall_clock_elapsed,
            "estimated_time_to_checkpoint": estimated_time_to_checkpoint,
        }
    
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Generate mock checkpoint data"""
        checkpoints = []
        
        for i in range(5):
            checkpoint = {
                "id": f"checkpoint_{i+1}",
                "name": f"Training Run {i+1}",
                "episode": 1000 + i * 500,
                "score": 2000 + i * 200,
                "created_at": datetime.now().isoformat(),
                "model_size": random.choice(["small", "medium", "large"]),
                "loss": 0.15 + i * 0.02,
                "file_size": f"{random.randint(50, 150)}MB"
            }
            checkpoints.append(checkpoint)
        
        return checkpoints 