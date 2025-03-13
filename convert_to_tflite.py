import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import numpy as np

def load_weights_bin(model, weights_bin_path):
    """Tenta carregar os pesos de um arquivo binário."""
    if not os.path.exists(weights_bin_path):
        raise FileNotFoundError(f"Arquivo de pesos '{weights_bin_path}' não encontrado.")

    try:
        weights = np.fromfile(weights_bin_path, dtype=np.float32)
        if len(weights) == 0:
            raise ValueError("O arquivo de pesos está vazio.")
        
        # Tenta carregar os pesos diretamente
        model.set_weights(weights)
        print("Pesos carregados com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar os pesos do binário: {e}")

def convert_model(model_json_path, weights_bin_path, output_path):
    """Converte um modelo Keras salvo em JSON + pesos binários para TFLite."""
    # Verifica a existência dos arquivos
    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Arquivo de modelo JSON '{model_json_path}' não encontrado.")

    # Carrega a arquitetura do modelo
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read().strip()  # Remove espaços extras

    if not model_json:
        raise ValueError("O arquivo model.json está vazio ou inválido.")

    try:
        model = model_from_json(model_json)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo a partir do JSON: {e}")

    # Tenta carregar os pesos
    try:
        load_weights_bin(model, weights_bin_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar os pesos: {e}")

    # Converter o modelo para TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
    except Exception as e:
        raise RuntimeError(f"Erro na conversão para TFLite: {e}")

    # Salvar o modelo convertido
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Modelo convertido salvo em: {output_path}")

if __name__ == "__main__":
    model_json_path = 'training/model/model.json'
    weights_bin_path = 'training/model/weights.bin'
    output_path = 'training_model.tflite'

    convert_model(model_json_path, weights_bin_path, output_path)
