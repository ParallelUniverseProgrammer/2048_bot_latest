# 📐 Model Studio – Mobile-First Visual Designer (Grounded, Step-by-Step Edition)  

Integrates tightly with your live Python stack:

- `app/training/ppo_trainer.py`  
- `app/training/training_manager.py`  
- `app/models/game_transformer.py`  
- `app/models/model_config.py`  
- `app/environment/gym_2048_env.py`

Scope = one **Model Studio** tab.  
Everything executes on a phone; desktop is irrelevant.

---

## 🎯 **IMPLEMENTATION STATUS - COMPLETED** ✅

**Week 0 Goals:** ✅ **FULLY COMPLETED**

### ✅ **Backend Implementation**
- ✅ **API Router**: Complete `design_router.py` with all endpoints
- ✅ **Data Models**: Pydantic models for validation and serialization
- ✅ **Code Generation**: Dynamic PyTorch model generation from JSON
- ✅ **Storage**: In-memory design storage with CRUD operations
- ✅ **Integration**: Seamless integration with existing TrainingManager

### ✅ **Frontend Implementation**
- ✅ **UI Component**: Mobile-first `ModelStudioTab.tsx` with responsive design
- ✅ **State Management**: Complete Zustand store with full state handling
- ✅ **Validation**: Real-time client-side validation via web worker
- ✅ **Block Palette**: Interactive component selection interface
- ✅ **Layout**: Optimized for 70%+ canvas space with mobile padding

### ✅ **Key Features Delivered**
- ✅ **Design Management**: Create, edit, and manage model designs
- ✅ **Real-time Validation**: Instant feedback on design validity
- ✅ **Parameter Estimation**: Automatic parameter and memory calculation
- ✅ **Model Compilation**: Dynamic code generation and compilation
- ✅ **Training Integration**: One-click training with existing infrastructure
- ✅ **Mobile Optimization**: Touch-friendly interface with proper spacing
- ✅ **Style Guide Compliance**: Consistent design system adherence

### ✅ **Files Created/Modified**
```
backend/
├── app/api/design_router.py          ✅ Complete API implementation
├── app/models/generated/__init__.py  ✅ Package for generated models
└── main.py                           ✅ Router integration

frontend/
├── src/stores/designStore.ts         ✅ Zustand state management
├── src/workers/design_worker.ts      ✅ Client-side validation
├── src/components/ModelStudioTab.tsx ✅ Main UI component
└── src/App.tsx                       ✅ Tab integration

tests/
└── test_model_studio.py              ✅ API testing script
```

### ✅ **Current Capabilities**
1. **Design Creation**: Create new model designs with custom names
2. **Component Addition**: Add transformer layers, MoE layers, embeddings
3. **Real-time Validation**: Instant feedback on design constraints
4. **Parameter Estimation**: Automatic calculation of model size
5. **Model Compilation**: Generate and compile PyTorch models
6. **Training Integration**: Start training with compiled models
7. **Mobile Interface**: Touch-optimized responsive design
8. **Error Handling**: Comprehensive validation and error display

**Status:** 🎉 **READY FOR WEEK 1 - CANVAS IMPLEMENTATION**

---

## 0  High-Level Flow  

1. User drags blocks → JSON \(ModelArchitecture\).  
2. `POST /api/designs/{id}/compile` turns JSON into a concrete `GeneratedModel`
   that extends `GameTransformer`.  
3. `POST /api/designs/{id}/train` spins up `TrainingManager` with the compiled
   model.  
4. TrainingManager streams metrics already expected by the FE.

Keep those four hops in mind; every section below maps to one of them.

---

## 1  New Files & Folders  

```
app/
├── api/
│   └── design_router.py          # new FastAPI router
├── models/
│   └── generated/
│       └── <design_id>.py        # auto-generated model
├── utils/
│   └── codegen.py                # architecture → source code
frontend/
├── src/
│   ├── stores/designStore.ts
│   ├── workers/design_worker.ts
│   └── ModelStudioTab.tsx
```

---

## 2  Backend API – Explicit Contract  

```python
# app/api/design_router.py  (pseudo-code)

router = APIRouter(prefix="/api/designs")

@router.post("", status_code=201)
def create_design() -> DesignMeta:
    """Create empty design and return id"""

@router.get("/{design_id}")
def fetch_design(design_id) -> ModelArchitecture:
    """Return stored JSON"""

@router.post("/{design_id}/validate")
def validate_design(design_id, body: ModelArchitecture) -> ValidationResult:
    """Fast rule-based validation (no compilation)"""

@router.post("/{design_id}/compile")
def compile_design(design_id) -> CompileResult:
    """
    1. Pull JSON from DB.
    2. Run generate_model_code(JSON) -> (source, config_dict, param_count).
    3. Write to app/models/generated/<id>.py
    4. Return import_path & param_count
    """

@router.post("/{design_id}/train", status_code=202)
async def train_design(design_id) -> dict:
    """
    1. import_module(f"app.models.generated.{id}")
    2. cfg = DynamicModelConfig.from_architecture(JSON)
    3. trainer = PPOTrainer(config=cfg)
    4. manager = TrainingManager(global_ws)
    5. manager.trainer = trainer
    6. manager.reset_to_fresh_model()
    7. manager.start()
    """
```

Schema objects (FastAPI `pydantic`):

```python
class ModelComponent(BaseModel):
    id: str
    type: str
    props: Dict[str, Union[int, float, str, bool]]

class ModelArchitecture(BaseModel):
    id: str
    name: str
    components: List[ModelComponent]
    edges: List[Tuple[str, str]]

class ValidationResult(BaseModel):
    valid: bool
    paramCount: float   # millions
    estimatedMemory: int  # MB
    errors: List[str] = []

class CompileResult(BaseModel):
    import_path: str
    paramCount: float
    config: Dict[str, Any]
```

DB helper (pseudo):

```python
def _save(id: str, doc: dict):
    db.model_designs.update_one({"_id": id}, {"$set": doc}, upsert=True)
```

---

## 3  Frontend Store & Worker Skeleton  

### 3.1 Zustand Store  

```typescript
// src/stores/designStore.ts
import create from "zustand"

export const useDesignStore = create<DesignStore>((set) => ({
  currentDesign: null,
  validation: null,
  paramCount: 0,
  estimatedMemory: 0,
  // mutate helpers
  setDesign: (d) => set({ currentDesign: d }),
  setValidation: (v) => set({ validation: v }),
}))
```

### 3.2 Web-Worker for Validation  

```typescript
// src/workers/design_worker.ts
self.onmessage = async (e) => {
  const design = e.data as ModelArchitecture
  const result = validate(design)          // see pseudo-rules below
  postMessage(result)
}

// Pure JS rules – must mirror Section 4.1
function validate(design) {
  const errors: string[] = []

  const dModel = design.meta?.d_model || 128
  if (dModel % 8 !== 0) errors.push("d_model must be multiple of 8")

  const nHeads = design.meta?.n_heads || 4
  if (dModel % nHeads !== 0) errors.push("n_heads must divide d_model")

  const nExperts = design.meta?.n_experts || 4
  if (nExperts > 8) errors.push("max 8 experts supported")

  return {
    valid: errors.length === 0,
    errors,
    paramCount: estimateParams(design),
    estimatedMemory: estimateMem(design),
  }
}
```

---

## 4  Week-by-Week Roadmap (Detailed)  

### Week 0 – “Hello, Blank Canvas”

- Backend  
  - Add `design_router` to `main.py`.
  - Stub DB functions.

- Frontend  
  - Create `ModelStudioTab.tsx` with a single “+ Add Block” button.
  - Hook worker; on every store update send design to worker.

Expected outcome: `curl -X POST /api/designs` returns id; UI shows empty grid.

---

### Week 1 – Touch Drag & Snap 🎯 **NEXT UP**

**Prerequisites:** ✅ Week 0 foundation complete - ready to implement canvas

Step-by-step:

1. Install libs  
   ```bash
   pnpm i react-konva react-dnd-touch-backend
   ```

2. Build Block component:  

   ```tsx
   const Block: FC<{ data: ModelComponent }> = ({ data }) => (
     <Group draggable>
       <Rect width={56} height={56} fill="#2D9CDB" cornerRadius={8} />
       <Text text={data.type} fontSize={10} width={56} align="center" />
     </Group>
   )
   ```

3. Canvas panning: pinch-zoom ↔ `stage.scale()`.

4. On `dragend`, update `currentDesign.components[index].props.xy`.

5. Persist store to `IndexedDB`:

   ```typescript
   subscribe((state) => save("design", state.currentDesign), { equalityFn: shallow })
   ```

**Current Foundation:** ✅ All backend APIs, frontend state management, and UI framework ready
**Canvas Space:** ✅ 70%+ screen real estate allocated and optimized for mobile
**Integration Points:** ✅ Store and worker ready to handle drag & drop events

---

### Week 2 – Real-Time Validation ✅ **COMPLETED**

1. ✅ Every `300 ms` debounce, post current design to worker.  
2. ✅ Worker replies `ValidationResult`; show a floating banner:
   - Green ✅ if `valid`.
   - Red ❌ if any error; list first three.

3. ✅ Auto-update `parameterCount` + `estimatedMemory` overlay bottom-right.

**Status:** ✅ Fully implemented in Week 0
**Features:** Real-time validation, error display, parameter estimation, memory calculation

Pseudo-formula:

```typescript
function estimateParams(design) {
  let params = 0
  for (const c of design.components) {
    switch (c.type) {
      case "embedding":
        params += 4 * 4 * c.props.d_model
        break
      case "transformer_layer":
        params += 2 * c.props.d_model ** 2
        break
      case "moe_layer":
        params += c.props.n_experts * c.props.d_model ** 2
        break
    }
  }
  return (params / 1e6).toFixed(2)
}
```

---

### Week 3 – Server-Side Compilation ✅ **COMPLETED**

Add `app/utils/codegen.py`.

**Status:** ✅ Fully implemented in Week 0
**Features:** Dynamic PyTorch model generation, parameter counting, configuration extraction

```python
# pseudo-code
TEMPLATE = """
from app.models.game_transformer import GameTransformer
import torch.nn as nn

class GeneratedModel(GameTransformer):
    def __init__(self, **cfg):
        super().__init__(**cfg)
{layers}

    def forward(self, x):
{forward_body}
"""

def generate_model_code(arch: dict) -> Tuple[str, dict, float]:
    layers = []
    forward = ["        h = x"]
    for i, comp in enumerate(arch["components"]):
        if comp["type"] == "transformer_layer":
            layers.append(f"        self.t{i} = nn.TransformerEncoderLayer("
                          f"d_model={comp['props']['d_model']}, "
                          f"nhead={comp['props']['n_heads']})")
            forward.append(f"        h = self.t{i}(h)")
        elif comp["type"] == "moe_layer":
            layers.append(f"        self.m{i} = nn.MoELayer("
                          f"d_model={comp['props']['d_model']}, "
                          f"n_experts={comp['props']['n_experts']})")
            forward.append(f"        h = self.m{i}(h)")
    forward.append("        return self.head(h)")
    code = TEMPLATE.format(layers="\n".join(layers),
                           forward_body="\n".join(forward))
    params = _count_params(code)            # helper
    cfg = {
        "d_model": arch["meta"]["d_model"],
        "n_heads": arch["meta"]["n_heads"],
        "n_layers": sum(1 for c in arch["components"]
                        if c["type"] == "transformer_layer"),
        "n_experts": max(c["props"]["n_experts"] for c in arch["components"]
                         if c["type"] == "moe_layer") if any(...) else 0,
    }
    return code, cfg, params
```

`compile_design` writes file, then:

```python
with open(path, "w") as f:
    f.write(source_code)
return {"import_path": f"app.models.generated.{id}",
        "paramCount": params, "config": cfg}
```

---

### Week 4 – One-Tap Train Hook ✅ **COMPLETED**

Backend function in router:

**Status:** ✅ Fully implemented in Week 0
**Features:** TrainingManager integration, dynamic trainer injection, non-blocking training start

```python
@router.post("/{design_id}/train")
async def train_design(design_id: str):
    arch = db.model_designs.find_one({"_id": design_id})
    mod = import_module(f"app.models.generated.{design_id}")
    model_cls = mod.GeneratedModel

    cfg = DynamicModelConfig.from_architecture(arch)
    trainer = PPOTrainer(config=cfg)
    manager = TrainingManager(global_ws_manager)   # already in memory
    manager.trainer = trainer
    manager.reset_to_fresh_model()                 # frees GPU if any
    manager.start()                                # non-blocking
    return {"run_id": f"run_{design_id}_{int(time.time())}"}
```

No edits to TrainingManager needed; it already supports external trainer
injection.

---

### Week 5 – MoE Heat-Map

Extend TrainingManager.build_metrics:

```python
# --- Roadmap Week 5 ---
moe_metrics = {
    "expert_usage": trainer_metrics["expert_usage"],
    "lb_reward": self._load_balancing_metrics[-1] if self._load_balancing_metrics else 0.0,
}
metrics["moe"] = moe_metrics
```

Frontend:

```tsx
// Inside BlockRenderer for MOE_LAYER
const usage = props.metrics?.moe?.expert_usage || []
const color = valueToHeat(usage[block.props.expertIndex])
<Rect fill={color} .../>
```

---

### Week 6 – Export Center

Router:

```python
@router.get("/{design_id}/export")
def export_model(design_id: str, fmt: str = Query("pt")):
    ckpt = checkpoints.latest_for_design(design_id)
    mod = import_module(f"app.models.generated.{design_id}")
    model = mod.GeneratedModel(**ckpt["config"])
    model.load_state_dict(torch.load(ckpt["path"])["model_state_dict"])
    dummy = torch.randn(1, 16, model.d_model)
    if fmt == "onnx":
        f = f"/tmp/{design_id}.onnx"
        torch.onnx.export(model, dummy, f, opset_version=15)
        return FileResponse(f, media_type="application/octet-stream")
    ...
```

---

### Week 7 – Advanced Blocks & Codegen Enhancements

Add block types:

- `LOAD_BALANCE_ROUTER`
- `AUX_LOSS_HEAD`
- `GRAD_CHECKPOINT`

Codegen updates:

```python
if comp["type"] == "load_balance_router":
    cfg["lb_reward_coef"] = comp["props"]["reward_coef"]
elif comp["type"] == "aux_loss_head":
    layers.append(f"        self.aux{i} = nn.Linear("
                  f"{comp['props']['d_model']}, 1)")
    forward.append(f"        aux_out = self.aux{i}(h.mean(1))")
    outputs.append("aux_out")
elif comp["type"] == "grad_checkpoint":
    forward_body = "torch.utils.checkpoint.checkpoint(self.t{}, h)".format(i)
```

---

### Week 8 – Performance & UI Polish  

1. Virtualize canvas at zoom < 0.4×:
   - Render single colored rectangles instead of full SVG labels.

2. Debounce pinch events to 60 fps:  

   ```tsx
   const raf = useRef(0)
   const onWheel = (e) => {
     cancelAnimationFrame(raf.current)
     raf.current = requestAnimationFrame(() => setScale(newScale))
   }
   ```

3. Long-press block → duplicate.  
4. Undo/redo: keep `past[]` `present` `future[]` stacks in Zustand.

---

### Week 9 – QA Handoff Prep  

- Add `npm run schema-lint` that validates every template JSON
  against `ModelArchitecture` schema.  
- Provide `scripts/dump_designs.py` to export all DB designs to
  `ci/artifacts/designs/*.json` for automated CI.

(No physical device matrix here.)

---

### Week 10 – Telemetry & Launch  

FE:

```typescript
analytics.track("compile_success", {
  designId,
  paramCount,
  time_ms: performance.now() - t0,
})
analytics.track("train_started", { designId })
```

Backend:

```python
router.middleware("http")
async def add_request_id(request, call_next):
    request.state.req_id = uuid4().hex
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.req_id
    return response
```

---

## 5  Changes to Existing Source Files  

### 5.1 `app/models/game_transformer.py`

```python
# Roadmap Week 3
class GameTransformer(nn.Module):
    ...
    @staticmethod
    def from_architecture(arch: dict) -> "GameTransformer":
        cfg = DynamicModelConfig.from_architecture(arch)
        mod = import_module(f"app.models.generated.{arch['id']}")
        return mod.GeneratedModel(**cfg.__dict__)
```

### 5.2 `app/models/model_config.py`

```python
# Roadmap Week 4
@classmethod
def from_architecture(cls, arch: dict) -> "DynamicModelConfig":
    return cls(
        d_model=arch["meta"]["d_model"],
        n_heads=arch["meta"]["n_heads"],
        n_experts=max(
            (c["props"]["n_experts"] for c in arch["components"]
             if c["type"] == "moe_layer"),
            default=0
        ),
        n_layers=sum(1 for c in arch["components"]
                     if c["type"] == "transformer_layer"),
    )
```

### 5.3 `app/training/ppo_trainer.py`

Ensure any generated model sets `self.latest_lb_loss`:

```python
# In calculate_load_balancing_reward()
self.model.latest_lb_loss = torch.tensor(lb_reward, device=self.device)
```

### 5.4 `app/training/training_manager.py`

Add helper:

```python
def adopt_external_trainer(self, trainer: PPOTrainer):
    self.trainer = trainer
    self.env_trainers = [trainer] * len(self.envs)
```

Used by Week 4 train endpoint.

---

## 6  Validation Rules Cheat-Sheet (mirrors worker & server)  

1. \(d_{\text{model}} \mod 8 = 0\).  
2. \(d_{\text{model}} / n_{\text{heads}} \in \mathbb{N}\).  
3. \(n_{\text{experts}} \le 8\).  
4. At least one `BOARD_INPUT` and one `ACTION_OUTPUT`.  
5. Graph must be acyclic and all nodes reachable from input.  

Failure → HTTP 422 with list of errors.

---

## 7  Definition of Done  

- Create new design, press Compile → server writes `generated/<id>.py`, returns
  param count.  
- Press Train → `TrainingManager` begins episodes and streams `training_update`
  with `moe.expert_usage` array.  
- Stop, Export as `.onnx`, file downloads and can be loaded via `onnxruntime`.  

All within a clean phone browser session; no desktop workflows involved.

---

## 8  Quick-Start for a Junior Dev  

```bash
# 1. backend
uvicorn app.main:app --reload
# 2. frontend
cd frontend && pnpm dev
# 3. open http://<phone-ip>:5173/model-studio
```

Cheat sheet:

1. Add “Transform-er” block → validates red if d\_model not multiple of 8.  
2. Tap ‘Compile’ → watch backend log `Writing generated/<id>.py`.  
3. Tap ‘Train’ → observe live score chart and MoE heat-map.

Welcome aboard.

---

## 📊 **PROGRESS SUMMARY**

### ✅ **Completed (Week 0)**
- **Backend Foundation**: Complete API with all endpoints
- **Frontend Foundation**: Mobile-first UI with state management
- **Validation System**: Real-time client-side validation
- **Code Generation**: Dynamic PyTorch model creation
- **Training Integration**: Seamless TrainingManager integration
- **Mobile Optimization**: Touch-friendly responsive design

### 🎯 **Next Priority (Week 1)**
- **Canvas Implementation**: Drag & drop interface with react-konva
- **Touch Interactions**: Pinch-zoom and mobile gesture support
- **Visual Design**: Block rendering and connection visualization
- **Persistence**: IndexedDB integration for design storage

### 📋 **Remaining Weeks (2-10)**
- **Week 5**: MoE Heat-Map visualization
- **Week 6**: Export Center for model formats
- **Week 7**: Advanced blocks and codegen enhancements
- **Week 8**: Performance optimization and UI polish
- **Week 9**: QA preparation and testing
- **Week 10**: Telemetry and launch preparation

### 🚀 **Current Status**
**Foundation:** ✅ **COMPLETE** - Ready for canvas implementation
**Next Step:** 🎯 **Week 1** - Touch drag & snap canvas interface
**Timeline:** On track for full implementation within roadmap timeframe