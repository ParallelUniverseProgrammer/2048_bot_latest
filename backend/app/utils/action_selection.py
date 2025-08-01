"""
Action selection utilities with fallback mechanism for invalid moves
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from copy import deepcopy

# Configure logging
logger = logging.getLogger(__name__)


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
    
    # Input validation
    if model is None:
        logger.error("Model is None, using random action selection")
        return _fallback_random_action(legal_actions)
    
    if state is None or len(state) == 0:
        logger.error("State is invalid, using random action selection")
        return _fallback_random_action(legal_actions)
    
    if not legal_actions:
        logger.warning("No legal actions available, using action 0")
        return 0, 0.0, None
    
    try:
        with torch.no_grad():
            # Convert state to tensor with error handling
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            except Exception as e:
                logger.error(f"Failed to convert state to tensor: {e}")
                return _fallback_random_action(legal_actions)
            
            # Get policy and value from model with error handling
            try:
                policy_logits, value = model(state_tensor)
                # Ensure logits are on the correct device to avoid CUDA/CPU mismatch
                policy_logits = policy_logits.to(device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "device" in str(e):
                    logger.error(f"CUDA/device error in model inference: {e}")
                    # Try moving everything to CPU as fallback
                    try:
                        state_tensor = state_tensor.cpu()
                        model_cpu = model.cpu()
                        policy_logits, value = model_cpu(state_tensor)
                        device = torch.device('cpu')
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed: {cpu_e}")
                        return _fallback_random_action(legal_actions)
                else:
                    logger.error(f"Model inference error: {e}")
                    return _fallback_random_action(legal_actions)
            except Exception as e:
                logger.error(f"Unexpected error in model inference: {e}")
                return _fallback_random_action(legal_actions)
            
            # Validate model output
            if policy_logits is None or torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                logger.error("Model returned invalid logits (NaN or Inf)")
                return _fallback_random_action(legal_actions)
            
            try:
                # Mask illegal actions
                action_mask = torch.full((4,), -float('inf'), device=device)
                action_mask[legal_actions] = 0.0
                masked_logits = policy_logits[0] + action_mask
                
                # Get action probabilities
                action_probs = F.softmax(masked_logits, dim=-1)
                
                # Validate probabilities
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    logger.error("Action probabilities contain NaN or Inf")
                    return _fallback_random_action(legal_actions)
                
                # Sort actions by probability (descending)
                sorted_probs, sorted_actions = torch.sort(action_probs, descending=True)
                
            except Exception as e:
                logger.error(f"Error in action masking/probability calculation: {e}")
                return _fallback_random_action(legal_actions)
            
            # Get attention weights for visualization if available
            attention_weights = None
            try:
                if hasattr(model, 'attention_weights') and model.attention_weights is not None:
                    attention_weights = model.attention_weights.clone()
            except Exception as e:
                logger.debug(f"Could not retrieve attention weights: {e}")
            
            # Try actions in order of probability until we find a valid one
            for attempt in range(min(max_attempts, len(legal_actions))):
                try:
                    action = sorted_actions[attempt].item()
                    action_prob = sorted_probs[attempt].item()
                    
                    # Skip if this action is not in legal actions (shouldn't happen due to masking)
                    if action not in legal_actions:
                        continue
                    
                    # Test if this action results in a valid move (same logic as _can_move)
                    try:
                        temp_game = deepcopy(env_game)
                        moved, reward = temp_game._move(action)
                        
                        # Check if the move was valid
                        if moved:
                            # Valid move found! Calculate log probability
                            action_dist = torch.distributions.Categorical(action_probs)
                            log_prob = action_dist.log_prob(torch.tensor(action).to(device))
                            
                            return action, log_prob.item(), attention_weights
                    except Exception as e:
                        logger.warning(f"Error testing action {action}: {e}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error in action selection attempt {attempt}: {e}")
                    continue
            
            # If no valid move found (shouldn't happen with proper legal_moves),
            # fall back to the first legal action
            fallback_action = legal_actions[0] if legal_actions else 0
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(fallback_action).to(device))
                return fallback_action, log_prob.item(), attention_weights
            except Exception as e:
                logger.error(f"Error in fallback action calculation: {e}")
                return fallback_action, 0.0, attention_weights
    
    except Exception as e:
        logger.error(f"Unexpected error in action selection: {e}")
        return _fallback_random_action(legal_actions)


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
    
    # Input validation
    if model is None:
        logger.error("Model is None, using random action selection")
        return _fallback_random_action_for_playback(legal_actions)
    
    if state is None or len(state) == 0:
        logger.error("State is invalid, using random action selection")
        return _fallback_random_action_for_playback(legal_actions)
    
    if not legal_actions:
        logger.warning("No legal actions available, using action 0")
        return 0, [0.25, 0.25, 0.25, 0.25], None
    
    try:
        with torch.no_grad():
            # Convert state to tensor with error handling
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            except Exception as e:
                logger.error(f"Failed to convert state to tensor: {e}")
                return _fallback_random_action_for_playback(legal_actions)
            
            # Get policy and value from model with error handling
            try:
                policy_logits, value = model(state_tensor)
                # Ensure logits are on the correct device to avoid CUDA/CPU mismatch
                policy_logits = policy_logits.to(device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "device" in str(e):
                    logger.error(f"CUDA/device error in model inference: {e}")
                    # Try moving everything to CPU as fallback
                    try:
                        state_tensor = state_tensor.cpu()
                        model_cpu = model.cpu()
                        policy_logits, value = model_cpu(state_tensor)
                        device = torch.device('cpu')
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed: {cpu_e}")
                        return _fallback_random_action_for_playback(legal_actions)
                else:
                    logger.error(f"Model inference error: {e}")
                    return _fallback_random_action_for_playback(legal_actions)
            except Exception as e:
                logger.error(f"Unexpected error in model inference: {e}")
                return _fallback_random_action_for_playback(legal_actions)
            
            # Validate model output
            if policy_logits is None or torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
                logger.error("Model returned invalid logits (NaN or Inf)")
                return _fallback_random_action_for_playback(legal_actions)
            
            try:
                # Mask illegal actions
                action_mask = torch.full((4,), -float('inf'), device=device)
                action_mask[legal_actions] = 0.0
                masked_logits = policy_logits[0] + action_mask
                
                # Get action probabilities
                action_probs = F.softmax(masked_logits, dim=-1)
                
                # Validate probabilities
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    logger.error("Action probabilities contain NaN or Inf")
                    return _fallback_random_action_for_playback(legal_actions)
                
                action_probs_list = action_probs.cpu().numpy().tolist()
                
                # Sort actions by probability (descending)
                sorted_probs, sorted_actions = torch.sort(action_probs, descending=True)
                
            except Exception as e:
                logger.error(f"Error in action masking/probability calculation: {e}")
                return _fallback_random_action_for_playback(legal_actions)
            
            # Get attention weights for visualization if available
            attention_weights = None
            try:
                if hasattr(model, 'attention_weights') and model.attention_weights is not None:
                    attention_weights = model.attention_weights.cpu().numpy()
            except Exception as e:
                logger.debug(f"Could not retrieve attention weights: {e}")
            
            # Try actions in order of probability until we find a valid one
            for attempt in range(min(4, len(legal_actions))):
                try:
                    action = sorted_actions[attempt].item()
                    action_prob = sorted_probs[attempt].item()
                    
                    # Skip if this action is not in legal actions (shouldn't happen due to masking)
                    if action not in legal_actions:
                        continue
                    
                    # Test if this action results in a valid move (same logic as _can_move)
                    try:
                        temp_game = deepcopy(env_game)
                        moved, reward = temp_game._move(action)
                        
                        # Check if the move was valid
                        if moved:
                            # Valid move found!
                            return action, action_probs_list, attention_weights
                    except Exception as e:
                        logger.warning(f"Error testing action {action}: {e}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error in action selection attempt {attempt}: {e}")
                    continue
            
            # If no valid move found (shouldn't happen with proper legal_moves),
            # fall back to the first legal action
            fallback_action = legal_actions[0] if legal_actions else 0
            return fallback_action, action_probs_list, attention_weights
    
    except Exception as e:
        logger.error(f"Unexpected error in action selection: {e}")
        return _fallback_random_action_for_playback(legal_actions)


def _fallback_random_action(legal_actions: List[int]) -> Tuple[int, float, Optional[torch.Tensor]]:
    """Fallback to random action selection when model fails"""
    import random
    
    if not legal_actions:
        return 0, 0.0, None
    
    action = random.choice(legal_actions)
    log_prob = -np.log(len(legal_actions))  # Uniform distribution log probability
    
    logger.warning(f"Using random fallback action: {action}")
    return action, log_prob, None


def _fallback_random_action_for_playback(legal_actions: List[int]) -> Tuple[int, List[float], Optional[np.ndarray]]:
    """Fallback to random action selection for playback when model fails"""
    import random
    
    if not legal_actions:
        return 0, [0.25, 0.25, 0.25, 0.25], None
    
    action = random.choice(legal_actions)
    
    # Create uniform probabilities for legal actions
    probs = [0.0, 0.0, 0.0, 0.0]
    uniform_prob = 1.0 / len(legal_actions)
    for legal_action in legal_actions:
        probs[legal_action] = uniform_prob
    
    logger.warning(f"Using random fallback action for playback: {action}")
    return action, probs, None 