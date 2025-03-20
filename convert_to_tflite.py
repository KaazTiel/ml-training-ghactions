import os
import subprocess

def convert_tfjs_to_tf(tfjs_model_dir, tf_model_path):
    """Converte um modelo TensorFlow.js para um modelo TensorFlow (Keras HDF5)."""
    command = [
        "tensorflowjs_converter",
        "--input_format=tfjs_layers_model",
        "--output_format=keras",
        tfjs_model_dir,
        tf_model_path
    ]
    subprocess.run(command, check=True)

def main():
    tfjs_model_dir = "training/model"  # Pasta onde estão model.json e weights.bin
    tf_model_path = "training/model/converted_model.h5"  # Salvar dentro da pasta model

    print("Convertendo TensorFlow.js para TensorFlow (Keras HDF5)...")
    convert_tfjs_to_tf(tfjs_model_dir, tf_model_path)
    print(f"Conversão concluída! Modelo salvo em: {tf_model_path}")

if __name__ == "__main__":
    main()