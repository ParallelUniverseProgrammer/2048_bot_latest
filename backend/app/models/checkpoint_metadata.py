"""
Checkpoint metadata management system
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint"""
    id: str
    nickname: str
    episode: int
    created_at: str
    training_duration: float  # seconds
    model_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    file_size: int  # bytes
    parent_checkpoint: Optional[str] = None
    tags: Optional[List[str]] = None
    absolute_path: Optional[str] = None  # New field for absolute path
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.absolute_path is None and hasattr(self, 'id'):
            # Fallback for legacy: construct absolute path
            self.absolute_path = str((Path(os.getenv('CHECKPOINTS_DIR', 'checkpoints')) / f"{self.id}.pt").resolve())

class CheckpointManager:
    """Manages checkpoint metadata and operations"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        print(f"[CheckpointManager] Using checkpoint_dir: {self.checkpoint_dir.resolve()}")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata_cache = {}
        self._load_all_metadata()
    
    def _infer_model_size_from_experts(self, n_experts: int) -> str:
        """Infer model size based on number of experts"""
        if n_experts <= 4:
            return 'small'
        elif n_experts <= 6:
            return 'medium'
        else:
            return 'large'
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get the checkpoint file path (absolute)"""
        return (self.checkpoint_dir / f"{checkpoint_id}.pt").resolve()
    
    def _get_metadata_path(self, checkpoint_id: str) -> Path:
        """Get the metadata file path (absolute)"""
        return (self.checkpoint_dir / f"{checkpoint_id}.json").resolve()
    
    def _load_all_metadata(self):
        """Load all existing metadata files into cache"""
        self.metadata_cache = {}
        
        # Load existing metadata files
        for metadata_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    # Add absolute_path if missing
                    checkpoint_id = data.get('id')
                    abs_path = str((self.checkpoint_dir / f"{checkpoint_id}.pt").resolve())
                    data['absolute_path'] = abs_path
                    metadata = CheckpointMetadata(**data)
                    self.metadata_cache[metadata.id] = metadata
            except Exception as e:
                print(f"Error loading metadata from {metadata_file}: {e}")
        
        # Check for checkpoint files without metadata (legacy compatibility)
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            checkpoint_id = checkpoint_file.stem
            if checkpoint_id not in self.metadata_cache:
                # Create metadata for legacy checkpoint
                metadata = self._create_legacy_metadata(checkpoint_id, checkpoint_file)
                if metadata:
                    self.metadata_cache[checkpoint_id] = metadata
                    self._save_metadata(metadata)
    
    def _create_legacy_metadata(self, checkpoint_id: str, checkpoint_file: Path) -> Optional[CheckpointMetadata]:
        """Create metadata for existing checkpoint files without metadata"""
        try:
            import torch
            
            # Load checkpoint to extract information
            checkpoint_data = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            
            # Extract episode number from filename or checkpoint data
            episode = checkpoint_data.get('episode_count', 0)
            if episode == 0:
                # Try to extract from filename
                parts = checkpoint_id.split('_')
                for i, part in enumerate(parts):
                    if part == 'episode' and i + 1 < len(parts):
                        try:
                            episode = int(parts[i + 1])
                            break
                        except ValueError:
                            pass
            
            # Get file stats
            file_stats = checkpoint_file.stat()
            created_at = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            file_size = file_stats.st_size
            
            # Extract performance metrics
            best_score = checkpoint_data.get('best_score', 0)
            loss_history = checkpoint_data.get('loss_history', [])
            score_history = checkpoint_data.get('score_history', [])
            
            # Calculate average score
            avg_score = 0
            if score_history:
                if isinstance(score_history[0], tuple):
                    avg_score = sum(score[1] for score in score_history[-100:]) / min(len(score_history), 100)
                else:
                    avg_score = sum(score_history[-100:]) / min(len(score_history), 100)
            
            # Extract model config
            config = checkpoint_data.get('config')
            model_config = {}
            if config:
                n_experts = getattr(config, 'n_experts', 6)
                inferred_size = self._infer_model_size_from_experts(n_experts)
                existing_size = getattr(config, 'model_size', 'unknown')
                # Use inferred size if model_size is missing or 'unknown'
                final_size = inferred_size if existing_size == 'unknown' else existing_size
                model_config = {
                    'model_size': final_size,
                    'learning_rate': 0.0003,  # Default, not stored in old checkpoints
                    'n_experts': n_experts,
                    'n_layers': getattr(config, 'n_layers', 6),
                    'd_model': getattr(config, 'd_model', 384),
                    'n_heads': getattr(config, 'n_heads', 8),
                }
            
            # Create metadata
            metadata = CheckpointMetadata(
                id=checkpoint_id,
                nickname=f"Episode {episode}",
                episode=episode,
                created_at=created_at,
                training_duration=0,  # Unknown for legacy checkpoints
                model_config=model_config,
                performance_metrics={
                    'best_score': best_score,
                    'avg_score': avg_score,
                    'final_loss': 0.0,  # Unknown for legacy checkpoints
                    'training_speed': 0.0,  # Unknown for legacy checkpoints
                    # Enhanced metrics (N/A for legacy)
                    'score_trend': 0.0,
                    'loss_trend': 0.0,
                    'max_tile_frequency': {},
                    'training_efficiency': {
                        'score_consistency': 0.0,
                        'loss_stability': 0.0,
                        'improvement_rate': 0.0,
                        'plateau_detection': 0.0
                    },
                },
                file_size=file_size,
                tags=['legacy']
            )
            
            return metadata
            
        except Exception as e:
            print(f"Error creating legacy metadata for {checkpoint_id}: {e}")
            return None
    
    def _save_metadata(self, metadata: CheckpointMetadata):
        """Save metadata to file"""
        metadata_path = self._get_metadata_path(metadata.id)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            print(f"Error saving metadata for {metadata.id}: {e}")
    
    def create_checkpoint_metadata(self, 
                                 checkpoint_id: str,
                                 episode: int,
                                 training_duration: float,
                                 model_config: Dict[str, Any],
                                 performance_metrics: Dict[str, Any],
                                 nickname: Optional[str] = None,
                                 parent_checkpoint: Optional[str] = None,
                                 tags: Optional[List[str]] = None) -> CheckpointMetadata:
        """Create new checkpoint metadata"""
        
        if nickname is None:
            nickname = f"Episode {episode}"
        
        if tags is None:
            tags = []
        
        # Add automatic tags based on performance
        if performance_metrics.get('best_score', 0) >= 2048:
            tags.append('2048_achieved')
        if performance_metrics.get('best_score', 0) >= 4096:
            tags.append('4096_achieved')
        if episode % 1000 == 0:
            tags.append('milestone')
        
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        file_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
        
        metadata = CheckpointMetadata(
            id=checkpoint_id,
            nickname=nickname,
            episode=episode,
            created_at=datetime.now().isoformat(),
            training_duration=training_duration,
            model_config=model_config,
            performance_metrics=performance_metrics,
            file_size=file_size,
            parent_checkpoint=parent_checkpoint,
            tags=tags,
            absolute_path=str(checkpoint_path)
        )
        
        self.metadata_cache[checkpoint_id] = metadata
        self._save_metadata(metadata)
        return metadata
    
    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        meta = self.metadata_cache.get(checkpoint_id)
        if meta and not meta.absolute_path:
            meta.absolute_path = str(self._get_checkpoint_path(checkpoint_id))
        return meta
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all checkpoints sorted by episode number, with absolute paths"""
        checkpoints = list(self.metadata_cache.values())
        for cp in checkpoints:
            if not cp.absolute_path:
                cp.absolute_path = str(self._get_checkpoint_path(cp.id))
        return sorted(checkpoints, key=lambda x: x.episode, reverse=True)
    
    def update_checkpoint_nickname(self, checkpoint_id: str, new_nickname: str) -> bool:
        """Update checkpoint nickname"""
        if checkpoint_id not in self.metadata_cache:
            return False
        
        metadata = self.metadata_cache[checkpoint_id]
        metadata.nickname = new_nickname
        self._save_metadata(metadata)
        return True
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint and its metadata"""
        if checkpoint_id not in self.metadata_cache:
            return False
        
        try:
            # Delete files
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            metadata_path = self._get_metadata_path(checkpoint_id)
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from cache
            del self.metadata_cache[checkpoint_id]
            return True
            
        except Exception as e:
            print(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    def checkpoint_exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists"""
        return checkpoint_id in self.metadata_cache
    
    def refresh_metadata_cache(self):
        """Refresh the metadata cache and update any 'unknown' model sizes"""
        self._load_all_metadata()
        
        # Update any existing checkpoints with 'unknown' model size
        updated_count = 0
        for checkpoint_id, metadata in self.metadata_cache.items():
            if metadata.model_config.get('model_size') == 'unknown':
                n_experts = metadata.model_config.get('n_experts', 6)
                inferred_size = self._infer_model_size_from_experts(n_experts)
                metadata.model_config['model_size'] = inferred_size
                self._save_metadata(metadata)
                updated_count += 1
        
        if updated_count > 0:
            print(f"Updated {updated_count} checkpoints with inferred model sizes")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about all checkpoints"""
        checkpoints = list(self.metadata_cache.values())
        
        if not checkpoints:
            return {
                'total_checkpoints': 0,
                'total_size': 0,
                'best_score': 0,
                'latest_episode': 0,
                'total_training_time': 0
            }
        
        total_size = sum(cp.file_size for cp in checkpoints)
        best_score = max(cp.performance_metrics.get('best_score', 0) for cp in checkpoints)
        latest_episode = max(cp.episode for cp in checkpoints)
        total_training_time = sum(cp.training_duration for cp in checkpoints)
        
        return {
            'total_checkpoints': len(checkpoints),
            'total_size': total_size,
            'best_score': best_score,
            'latest_episode': latest_episode,
            'total_training_time': total_training_time
        } 