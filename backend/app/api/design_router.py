from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Tuple, Union, Any, Optional
import time
import uuid
from importlib import import_module

router = APIRouter(prefix="/api/designs")

# Pydantic models for API contracts
class ModelComponent(BaseModel):
    id: str
    type: str
    props: Dict[str, Union[int, float, str, bool]]

class ModelArchitecture(BaseModel):
    id: str
    name: str
    components: List[ModelComponent]
    edges: List[Tuple[str, str]]
    meta: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    valid: bool
    paramCount: float  # millions
    estimatedMemory: int  # MB
    errors: List[str] = []

class CompileResult(BaseModel):
    import_path: str
    paramCount: float
    config: Dict[str, Any]

class DesignMeta(BaseModel):
    id: str
    name: str
    created_at: float
    updated_at: float

# In-memory storage for designs (replace with database in production)
designs_db: Dict[str, ModelArchitecture] = {}

@router.post("", status_code=201)
def create_design() -> DesignMeta:
    """Create empty design and return id"""
    design_id = f"design_{int(time.time())}"
    design = ModelArchitecture(
        id=design_id,
        name="Untitled Design",
        components=[],
        edges=[],
        meta={
            "d_model": 128,
            "n_heads": 4,
            "n_experts": 4
        }
    )
    designs_db[design_id] = design
    
    return DesignMeta(
        id=design_id,
        name=design.name,
        created_at=time.time(),
        updated_at=time.time()
    )

@router.get("/{design_id}")
def fetch_design(design_id: str) -> ModelArchitecture:
    """Return stored JSON"""
    if design_id not in designs_db:
        raise HTTPException(status_code=404, detail="Design not found")
    return designs_db[design_id]

@router.put("/{design_id}")
def update_design(design_id: str, design: ModelArchitecture) -> ModelArchitecture:
    """Update design"""
    if design_id not in designs_db:
        raise HTTPException(status_code=404, detail="Design not found")
    
    design.id = design_id  # Ensure ID matches
    designs_db[design_id] = design
    return design

@router.post("/{design_id}/validate")
def validate_design(design_id: str, body: ModelArchitecture) -> ValidationResult:
    """Fast rule-based validation (no compilation)"""
    errors = []
    
    # Extract meta parameters with defaults
    meta = body.meta or {}
    d_model = meta.get("d_model", 128)
    n_heads = meta.get("n_heads", 4)
    n_experts = meta.get("n_experts", 4)
    
    # Validation Rule 1: d_model must be multiple of 8
    if d_model % 8 != 0:
        errors.append("d_model must be multiple of 8")
    
    # Validation Rule 2: n_heads must divide d_model
    if d_model % n_heads != 0:
        errors.append("n_heads must divide d_model")
    
    # Validation Rule 3: max 8 experts supported
    if n_experts > 8:
        errors.append("max 8 experts supported")
    
    # Validation Rule 4: At least one BOARD_INPUT and one ACTION_OUTPUT
    has_board_input = any(c.type == "BOARD_INPUT" for c in body.components)
    has_action_output = any(c.type == "ACTION_OUTPUT" for c in body.components)
    
    if not has_board_input:
        errors.append("At least one BOARD_INPUT component required")
    
    if not has_action_output:
        errors.append("At least one ACTION_OUTPUT component required")
    
    # Validation Rule 5: Graph must be acyclic and all nodes reachable from input
    if body.components:
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            outgoing_edges = [edge for edge in body.edges if edge[0] == node]
            for _, to in outgoing_edges:
                if has_cycle(to):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check for cycles
        for component in body.components:
            if component.id not in visited and has_cycle(component.id):
                errors.append("Graph contains cycles")
                break
    
    # Calculate parameter count and memory estimate
    param_count = _estimate_params(body)
    estimated_memory = _estimate_memory(body)
    
    return ValidationResult(
        valid=len(errors) == 0,
        paramCount=param_count,
        estimatedMemory=estimated_memory,
        errors=errors
    )

@router.post("/{design_id}/compile")
def compile_design(design_id: str) -> CompileResult:
    """
    1. Pull JSON from DB.
    2. Run generate_model_code(JSON) -> (source, config_dict, param_count).
    3. Write to app/models/generated/<id>.py
    4. Return import_path & param_count
    """
    if design_id not in designs_db:
        raise HTTPException(status_code=404, detail="Design not found")
    
    design = designs_db[design_id]
    
    # Validate first
    validation = validate_design(design_id, design)
    if not validation.valid:
        raise HTTPException(status_code=422, detail="Design validation failed")
    
    try:
        # Generate model code
        source_code, config_dict, param_count = _generate_model_code(design)
        
        # Write to file
        import os
        generated_dir = os.path.join(os.path.dirname(__file__), "..", "models", "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        file_path = os.path.join(generated_dir, f"{design_id}.py")
        with open(file_path, "w") as f:
            f.write(source_code)
        
        return CompileResult(
            import_path=f"app.models.generated.{design_id}",
            paramCount=param_count,
            config=config_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compilation failed: {str(e)}")

@router.post("/{design_id}/train", status_code=202)
async def train_design(design_id: str) -> dict:
    """
    1. import_module(f"app.models.generated.{id}")
    2. cfg = DynamicModelConfig.from_architecture(JSON)
    3. trainer = PPOTrainer(config=cfg)
    4. manager = TrainingManager(global_ws)
    5. manager.trainer = trainer
    6. manager.reset_to_fresh_model()
    7. manager.start()
    """
    if design_id not in designs_db:
        raise HTTPException(status_code=404, detail="Design not found")
    
    try:
        # Import the generated model
        module_path = f"app.models.generated.{design_id}"
        mod = import_module(module_path)
        model_cls = getattr(mod, "GeneratedModel")
        
        # Get design for config
        design = designs_db[design_id]
        
        # Create config from architecture
        cfg = _create_model_config(design)
        
        # Import training components
        from app.training.ppo_trainer import PPOTrainer
        from app.training.training_manager import TrainingManager
        from app.api.websocket_manager import global_ws_manager
        
        # Create trainer and manager
        trainer = PPOTrainer(config=cfg)
        manager = TrainingManager(global_ws_manager)
        manager.trainer = trainer
        manager.reset_to_fresh_model()
        manager.start()
        
        return {
            "run_id": f"run_{design_id}_{int(time.time())}",
            "status": "started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")

# Helper functions
def _estimate_params(design: ModelArchitecture) -> float:
    """Estimate parameter count in millions"""
    params = 0
    d_model = design.meta.get("d_model", 128) if design.meta else 128
    
    for component in design.components:
        if component.type == "BOARD_INPUT":
            params += 4 * 4 * d_model  # 4x4 board to d_model embedding
        elif component.type == "TRANSFORMER_LAYER":
            params += 2 * d_model ** 2  # Self-attention + FFN
        elif component.type == "MOE_LAYER":
            n_experts = component.props.get("n_experts", 4)
            params += n_experts * d_model ** 2  # Multiple experts
        elif component.type == "ACTION_OUTPUT":
            params += d_model * 4  # d_model to 4 actions
        elif component.type == "VALUE_HEAD":
            params += d_model * 1  # d_model to 1 value
    
    return round((params / 1e6) * 100) / 100  # Return in millions, 2 decimal places

def _estimate_memory(design: ModelArchitecture) -> int:
    """Estimate memory usage in MB"""
    param_count = _estimate_params(design)
    d_model = design.meta.get("d_model", 128) if design.meta else 128
    
    # Rough estimation: 4 bytes per parameter + activation memory
    param_memory = param_count * 1e6 * 4  # bytes
    activation_memory = d_model * 16 * 4  # batch_size * sequence_length * d_model * 4 bytes
    
    return int((param_memory + activation_memory) / (1024 * 1024))  # Convert to MB

def _generate_model_code(design: ModelArchitecture) -> tuple[str, dict, float]:
    """Generate Python source code for the model"""
    template = '''from app.models.game_transformer import GameTransformer
import torch.nn as nn

class GeneratedModel(GameTransformer):
    def __init__(self, **cfg):
        super().__init__(**cfg)
{layers}

    def forward(self, x):
{forward_body}
'''
    
    layers = []
    forward_body = ["        h = x"]
    
    for i, component in enumerate(design.components):
        if component.type == "TRANSFORMER_LAYER":
            d_model = component.props.get("d_model", 128)
            n_heads = component.props.get("n_heads", 4)
            layers.append(f"        self.t{i} = nn.TransformerEncoderLayer("
                         f"d_model={d_model}, "
                         f"nhead={n_heads})")
            forward_body.append(f"        h = self.t{i}(h)")
        elif component.type == "MOE_LAYER":
            d_model = component.props.get("d_model", 128)
            n_experts = component.props.get("n_experts", 4)
            layers.append(f"        self.m{i} = nn.MoELayer("
                         f"d_model={d_model}, "
                         f"n_experts={n_experts})")
            forward_body.append(f"        h = self.m{i}(h)")
        elif component.type == "BOARD_INPUT":
            d_model = component.props.get("d_model", 128)
            layers.append(f"        self.input_embed = nn.Linear(16, {d_model})")
            forward_body.append(f"        h = self.input_embed(x.view(x.size(0), -1))")
        elif component.type == "ACTION_OUTPUT":
            d_model = component.props.get("d_model", 128)
            layers.append(f"        self.action_head = nn.Linear({d_model}, 4)")
            forward_body.append(f"        action_logits = self.action_head(h)")
        elif component.type == "VALUE_HEAD":
            d_model = component.props.get("d_model", 128)
            layers.append(f"        self.value_head = nn.Linear({d_model}, 1)")
            forward_body.append(f"        value = self.value_head(h.mean(1))")
    
    forward_body.append("        return action_logits, value")
    
    code = template.format(
        layers="\n".join(layers),
        forward_body="\n".join(forward_body)
    )
    
    # Create config dict
    meta = design.meta or {}
    config_dict = {
        "d_model": meta.get("d_model", 128),
        "n_heads": meta.get("n_heads", 4),
        "n_layers": sum(1 for c in design.components if c.type == "TRANSFORMER_LAYER"),
        "n_experts": max((c.props.get("n_experts", 4) for c in design.components 
                         if c.type == "MOE_LAYER"), default=0),
    }
    
    param_count = _estimate_params(design)
    
    return code, config_dict, param_count

def _create_model_config(design: ModelArchitecture) -> dict:
    """Create model configuration from architecture"""
    meta = design.meta or {}
    return {
        "d_model": meta.get("d_model", 128),
        "n_heads": meta.get("n_heads", 4),
        "n_layers": sum(1 for c in design.components if c.type == "TRANSFORMER_LAYER"),
        "n_experts": max((c.props.get("n_experts", 4) for c in design.components 
                         if c.type == "MOE_LAYER"), default=0),
    } 