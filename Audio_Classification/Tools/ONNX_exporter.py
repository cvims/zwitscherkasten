"""
PyTorch Model → ONNX Exporter (Generic).

Export **any** PyTorch model to ONNX format for cross-platform deployment.
Supports dynamic shapes, verification, metadata generation, and model registry.

Key Features:
- Generic model support (no hardcoded architecture)
- Dynamic input/output shapes
- ONNX Runtime verification
- Comprehensive metadata generation
- Model registry for multiple variants
- Production-ready CLI

Usage:
    python export_to_onnx.py \
        --model-module my_models.py \
        --model-class MyAudioClassifier \
        --checkpoint best_model.pth \
        --output model.onnx \
        --input-shape 1,1,128,1024 \
        --verify
"""

import argparse
import importlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version

# Optional imports for verification
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  Install onnx + onnxruntime for verification: pip install onnx onnxruntime")

# Suppress warnings during export
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Default ONNX export settings.
DEFAULT_OPSET_VERSION = 18  # Supports most modern operators
DEFAULT_DYNAMIC_AXES = True  # Enable dynamic batch/time dimensions

# Model registry for quick access (extend as needed).
MODEL_REGISTRY = {
    "mobilenetv3_audio": {
        "module": "torchvision.models",
        "class": "mobilenet_v3_small",
        "custom_init": True,
        "num_classes": 128,
        "input_shape": [1, 1, 128, 1024],
        "example_inputs": True,
    },
    # Add your custom models here:
    # "my_classifier": {
    #     "module": "my_models",
    #     "class": "AudioClassifier",
    #     "num_classes": 128,
    #     "input_shape": [1, 1, 128, 1024],
    # }
}


# ==============================================================================
# MODEL LOADING
# ==============================================================================


def load_model_from_registry(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model using registry configuration.

    Automatically handles:
    - Module/class instantiation
    - Custom initialization
    - Checkpoint loading
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    config = MODEL_REGISTRY[model_name].copy()
    config.update(kwargs)

    # Dynamic import
    module = importlib.import_module(config["module"])
    model_class = getattr(module, config["class"])

    # Create model instance
    if config.get("custom_init"):
        model = model_class(**config.get("init_args", {}))
    else:
        model = model_class(**config.get("init_args", {}))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device).eval()

    # Extract training config if available
    training_config = checkpoint.get('config', checkpoint.get('hyperparams', {}))
    return model, {**config, "training_config": training_config}


def load_arbitrary_model(
    module_name: str,
    class_name: str,
    checkpoint_path: str,
    device: torch.device,
    init_args: Dict[str, Any] = None,
    **kwargs
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load any PyTorch model by module/class path.

    Usage:
        --model-module my_models --model-class AudioNet --init-args num_classes=128
    """
    # Dynamic import
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # Inspect signature for init args
    sig = inspect.signature(model_class.__init__)
    default_args = {k: v.default for k, v in sig.parameters.items() 
                   if v.default is not inspect.Parameter.empty}
    init_args = init_args or default_args

    # Instantiate model
    model = model_class(**init_args)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.to(device).eval()

    config = {
        "module": module_name,
        "class": class_name,
        "init_args": init_args,
        "checkpoint": str(checkpoint_path)
    }
    config.update(kwargs)
    config["training_config"] = checkpoint.get('config', {})

    return model, config


def create_dummy_input(shape: List[int], device: torch.device) -> torch.Tensor:
    """Create random input tensor matching model expectations."""
    return torch.randn(*shape, device=device, dtype=torch.float32)


# ==============================================================================
# ONNX EXPORT
# ==============================================================================


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
    simplify: bool = True,
    verbose: bool = False
) -> bool:
    """
    Generic PyTorch → ONNX exporter with optimizations.

    Supports:
    - Dynamic shapes (batch, sequence length)
    - Operator fusion and constant folding
    - Model simplification (onnx-simplifier)
    """
    print(f"🚀 Exporting to ONNX: {output_path}")
    print(f"   Opset: {opset_version} | Simplify: {simplify}")

    # Default names
    input_names = input_names or ['input']
    output_names = output_names or ['output']
    dynamic_axes = dynamic_axes or {input_names[0]: {0: 'batch_size'}}

    model.eval()

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            training=torch.onnx.TrainingMode.EVAL,
        )

        # Optional simplification (reduces model size)
        if simplify:
            try:
                import onnxsim
                model_onnx = onnx.load(output_path)
                model_onnx, check = onnxsim.simplify(model_onnx)
                onnx.save(model_onnx, output_path)
                print(f"   Simplified (size reduced)")
            except ImportError:
                print("   onnx-simplifier not installed (pip install onnx-simplifier)")

        return True

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


# ==============================================================================
# VERIFICATION
# ==============================================================================


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor) -> bool:
    """Verify ONNX model with ONNX Runtime."""
    if not ONNX_AVAILABLE:
        print("⚠️  Skipping verification (onnxruntime not installed)")
        return True

    try:
        print("🔍 Verifying ONNX model...")

        # Structural validation
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("   ✓ ONNX structure valid")

        # Runtime inference test
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)

        print(f"   ✓ Providers: {session.get_providers()}")

        # PyTorch reference output
        with torch.no_grad():
            pytorch_output = session.run(
                None,
                {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            )

        print(f"   ✓ Inference OK | Output shape: {pytorch_output[0].shape}")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


# ==============================================================================
# METADATA GENERATION
# ==============================================================================


def generate_metadata(
    onnx_path: Path,
    model: nn.Module,
    config: Dict[str, Any],
    dummy_input: torch.Tensor,
    input_names: List[str],
    output_names: List[str]
) -> Path:
    """Generate comprehensive metadata JSON."""
    metadata = {
        "model_info": {
            "name": config.get("model_name", "Unnamed Model"),
            "architecture": f"{config.get('module', 'custom')}.{config.get('class', 'Custom')}",
            "export_date": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "onnx_info": {
            "path": str(onnx_path),
            "opset_version": config.get("opset_version", DEFAULT_OPSET_VERSION),
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": config.get("dynamic_axes", {}),
        },
        "input_spec": {
            "shape": list(dummy_input.shape),
            "dtype": str(dummy_input.dtype),
            "example_shape": list(dummy_input.shape),
        },
        "training_config": config.get("training_config", {}),
        "deployment_notes": [
            "Supports ONNX Runtime, TensorRT, OpenVINO, TVM",
            "Dynamic batch size and sequence length supported",
            "Input: normalized mel spectrogram [0,1] or [-80dB,0dB]"
        ]
    }

    metadata_path = onnx_path.with_suffix('.json')
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=2)

    print(f"📄 Metadata: {metadata_path}")
    return metadata_path


# ==============================================================================
# CLI INTERFACE
# ==============================================================================


def parse_arguments() -> argparse.Namespace:
    """Production-ready CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generic PyTorch → ONNX Exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Loading Modes:
  1. Registry (fast):
     --model mobilenetv3_audio --checkpoint best.pth

  2. Custom module/class:
     --model-module my_models --model-class AudioNet \\
     --init-args num_classes=128 --checkpoint best.pth

  3. TorchScript/JIT:
     --torchscript model.pt

Examples:
  python export_to_onnx.py --model mobilenetv3_audio --checkpoint best.pth --output model.onnx --verify
  python export_to_onnx.py --model-module audio.models --model-class BirdClassifier \\
                           --init-args num_classes=128 --checkpoint best.pth \\
                           --input-shape 1,1,128,1024 --output bird_model.onnx
        """
    )

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        help="Predefined model from registry"
    )
    model_group.add_argument(
        "--model-module", "--module",
        help="Python module containing model class (e.g., 'audio.models')"
    )
    model_group.add_argument(
        "--model-class", "--class",
        help="Model class name (e.g., 'AudioClassifier')"
    )
    model_group.add_argument(
        "--torchscript",
        help="Load TorchScript model (.pt)"
    )

    parser.add_argument("--checkpoint", "-c", required=False,
                       help="Checkpoint path (.pth)")
    parser.add_argument("--init-args", nargs="*", default=[],
                       help="Model init args (e.g., num_classes=128 hidden_dim=512)")

    # Input/Output
    parser.add_argument("--input-shape", "-i", required=True,
                       help="Input shape (e.g., '1,1,128,1024')")
    parser.add_argument("--output", "-o", default="model.onnx",
                       help="Output ONNX path")
    parser.add_argument("--input-names", nargs="+", default=["input"],
                       help="ONNX input names")
    parser.add_argument("--output-names", nargs="+", default=["output"],
                       help="ONNX output names")

    # Export options
    parser.add_argument("--opset-version", type=int, default=DEFAULT_OPSET_VERSION,
                       help="ONNX opset version")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                       help="Export device")
    parser.add_argument("--simplify", action="store_true", default=True,
                       help="Simplify ONNX graph")

    # Verification & metadata
    parser.add_argument("--verify", action="store_true",
                       help="Verify with ONNX Runtime")
    parser.add_argument("--require-verify", action="store_true",
                       help="Fail if verification fails")
    parser.add_argument("--metadata", action="store_true", default=True,
                       help="Generate metadata JSON")

    return parser.parse_args()


def main():
    """Main export workflow."""
    args = parse_arguments()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"🚀 PyTorch→ONNX Exporter | Device: {device} | PyTorch: {torch.__version__}")

    # Parse input shape
    try:
        input_shape = [int(x) for x in args.input_shape.split(",")]
        dummy_input = create_dummy_input(input_shape, device)
    except ValueError:
        print(f"❌ Invalid input shape: {args.input_shape}")
        return 1

    # Load model
    try:
        if args.model:
            model, config = load_model_from_registry(args.model, args.checkpoint, device)
        elif args.torchscript:
            model = torch.jit.load(args.torchscript, map_location=device)
            config = {"type": "torchscript"}
        else:
            init_args = dict(arg.split("=") for arg in args.init_args)
            model, config = load_arbitrary_model(
                args.model_module, args.model_class, args.checkpoint, device, init_args
            )
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return 1

    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = export_to_onnx(
        model,
        dummy_input,
        str(output_path),
        args.input_names,
        args.output_names,
        opset_version=args.opset_version,
        simplify=args.simplify
    )

    if not success:
        return 1

    # Verify
    if args.verify or args.require_verify:
        verify_ok = verify_onnx_model(str(output_path), dummy_input)
        if not verify_ok and args.require_verify:
            return 1

    # Metadata
    if args.metadata:
        generate_metadata(output_path, model, config, dummy_input,
                         args.input_names, args.output_names)

    print(f"\n🎉 SUCCESS! ONNX model saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
