"""
Action selection utilities with fallback mechanism for invalid moves
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from copy import deepcopy


def select_action_with_fallback(
    model: torch.nn.Module,
    state: np.ndarray,
    legal_actions: List[int],
    env_game,
    device: torch.device,
    sample_action: bool = True,
    max_attempts: int = 4
) -> Tuple[int, float, Optional[torch.Tensor]]:
    """
    Select action with fallback mechanism for invalid moves.
    
    If the top predicted action results in no board state change (invalid move),
    try the next most probable action until a valid move is found.
    
    Args:
        model: The neural network model
        state: Current board state as numpy array
        legal_actions: List of legal actions from game.legal_moves()
        env_game: The game environment object to test moves
        device: PyTorch device
        sample_action: If True, sample from distribution; if False, use argmax
        max_attempts: Maximum number of actions to try
        
    Returns:
        Tuple of (action, log_prob, attention_weights)
        - action: The selected valid action
        - log_prob: Log probability of the selected action
        - attention_weights: Attention weights for visualization (if available)
    """
    
    with torch.no_grad():
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get policy and value from model
        policy_logits, value = model(state_tensor)
        
        # Mask illegal actions
        action_mask = torch.full((4,), -float('inf'), device=device)
        action_mask[legal_actions] = 0.0
        masked_logits = policy_logits[0] + action_mask
        
        # Get action probabilities
        action_probs = F.softmax(masked_logits, dim=-1)
        
        # Sort actions by probability (descending)
        sorted_probs, sorted_actions = torch.sort(action_probs, descending=True)
        
        # Get attention weights for visualization if available
        attention_weights = None
        if hasattr(model, 'attention_weights') and model.attention_weights is not None:
            attention_weights = model.attention_weights.clone()
        
        # Try actions in order of probability until we find a valid one
        for attempt in range(min(max_attempts, len(legal_actions))):
            action = sorted_actions[attempt].item()
            action_prob = sorted_probs[attempt].item()
            
            # Skip if this action is not in legal actions (shouldn't happen due to masking)
            if action not in legal_actions:
                continue
            
            # Test if this action results in a valid move (same logic as _can_move)
            temp_game = deepcopy(env_game)
            moved, reward = temp_game._move(action)
            
            # Check if the move was valid
            if moved:
                # Valid move found! Calculate log probability
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action, device=device))
                
                return action, log_prob.item(), attention_weights
        
        # If no valid move found (shouldn't happen with proper legal_moves),
        # fall back to the first legal action
        fallback_action = legal_actions[0] if legal_actions else 0
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(torch.tensor(fallback_action, device=device))
        
        return fallback_action, log_prob.item(), attention_weights


def select_action_with_fallback_for_playback(
    model: torch.nn.Module,
    state: np.ndarray,
    legal_actions: List[int],
    env_game,
    device: torch.device,
    deterministic: bool = True
) -> Tuple[int, List[float], Optional[np.ndarray]]:
    """
    Select action with fallback for playback/inference (deterministic by default).
    
    Args:
        model: The neural network model
        state: Current board state as numpy array
        legal_actions: List of legal actions from game.legal_moves()
        env_game: The game environment object to test moves
        device: PyTorch device
        deterministic: If True, use argmax; if False, sample from distribution
        
    Returns:
        Tuple of (action, action_probs_list, attention_weights)
        - action: The selected valid action
        - action_probs_list: List of action probabilities [up, down, left, right]
        - attention_weights: Attention weights for visualization (if available)
    """
    
    with torch.no_grad():
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get policy and value from model
        policy_logits, value = model(state_tensor)
        
        # Mask illegal actions
        action_mask = torch.full((4,), -float('inf'), device=device)
        action_mask[legal_actions] = 0.0
        masked_logits = policy_logits[0] + action_mask
        
        # Get action probabilities
        action_probs = F.softmax(masked_logits, dim=-1)
        action_probs_list = action_probs.cpu().numpy().tolist()
        
        # Sort actions by probability (descending)
        sorted_probs, sorted_actions = torch.sort(action_probs, descending=True)
        
        # Get attention weights for visualization if available
        attention_weights = None
        if hasattr(model, 'attention_weights') and model.attention_weights is not None:
            attention_weights = model.attention_weights.cpu().numpy()
        
        # Try actions in order of probability until we find a valid one
        for attempt in range(min(4, len(legal_actions))):
            action = sorted_actions[attempt].item()
            action_prob = sorted_probs[attempt].item()
            
            # Skip if this action is not in legal actions (shouldn't happen due to masking)
            if action not in legal_actions:
                continue
            
            # Test if this action results in a valid move (same logic as _can_move)
            temp_game = deepcopy(env_game)
            moved, reward = temp_game._move(action)
            
            # Check if the move was valid
            if moved:
                # Valid move found!
                return action, action_probs_list, attention_weights
        
        # If no valid move found (shouldn't happen with proper legal_moves),
        # fall back to the first legal action
        fallback_action = legal_actions[0] if legal_actions else 0
        return fallback_action, action_probs_list, attention_weights 