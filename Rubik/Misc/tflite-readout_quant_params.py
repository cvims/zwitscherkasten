#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tflite_inspect.py /path/to/model.tflite")
        sys.exit(2)

    model_path = sys.argv[1]

    # Try tflite_runtime first (lightweight), fallback to tensorflow if needed
    Interpreter = None
    backend = None

    try:
        from tflite_runtime.interpreter import Interpreter as I
        Interpreter = I
        backend = "tflite_runtime"
    except Exception:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            backend = "tensorflow"
        except Exception as e:
            print("ERROR: Could not import tflite_runtime or tensorflow.")
            print("Install one of them, e.g.:")
            print("  pip3 install --user tflite-runtime")
            print("or (heavier):")
            print("  pip3 install --user tensorflow")
            print(f"\nDetails: {e}")
            sys.exit(1)

    print(f"Using backend: {backend}")
    print(f"Model: {model_path}")

    itp = Interpreter(model_path=model_path)
    itp.allocate_tensors()

    print("\n=== INPUTS ===")
    for i, d in enumerate(itp.get_input_details()):
        name = d.get("name")
        shape = d.get("shape")
        dtype = d.get("dtype")
        q = d.get("quantization", None)
        print(f"[{i}] name={name} shape={shape} dtype={dtype} quant={q}")

    print("\n=== OUTPUTS ===")
    for i, d in enumerate(itp.get_output_details()):
        name = d.get("name")
        shape = d.get("shape")
        dtype = d.get("dtype")
        q = d.get("quantization", None)
        print(f"[{i}] name={name} shape={shape} dtype={dtype} quant={q}")

if __name__ == "__main__":
    main()
