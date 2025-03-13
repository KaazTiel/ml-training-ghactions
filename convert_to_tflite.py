import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
import os

def load_weights_bin(model, weights_bin_path):
    """Carrega pesos do modelo a partir de um arquivo binário."""
    if not os.path.exists(weights_bin_path):
        raise FileNotFoundError(f"Arquivo de pesos '{weights_bin_path}' não encontrado.")

    try:
        weights = np.fromfile(weights_bin_path, dtype=np.float32)
        if len(weights) == 0:
            raise ValueError("O arquivo de pesos está vazio.")
        
        # Define os pesos no modelo
        model.set_weights([weights])
        print("Pesos carregados com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar os pesos do binário: {e}")

def convert_model(output_path):
    """Converte um modelo salvo em Keras (`.h5`) ou JSON + pesos binários (`.bin`) para TFLite."""
    
    # Diretórios esperados
    model_json_path = 'training/model/model.json'
    weights_bin_path = 'training/model/weights.bin'
    keras_model_path = 'training/model/model.h5'

    # Verifica se o modelo está salvo no formato Keras
    if os.path.exists(keras_model_path):
        print(f"Modelo encontrado em formato Keras: {keras_model_path}")
        try:
            model = load_model(keras_model_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o modelo Keras: {e}")

    # Caso contrário, verifica se está no formato JSON + binário
    elif os.path.exists(model_json_path) and os.path.exists(weights_bin_path):
        print(f"Modelo encontrado em formato JSON + pesos binários: {model_json_path}, {weights_bin_path}")
        
        # Carrega a arquitetura do modelo
        try:
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read().strip()
            if not model_json:
                raise ValueError("O arquivo model.json está vazio ou inválido.")

            model = model_from_json(model_json)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o modelo a partir do JSON: {e}")

        # Carrega os pesos
        try:
            load_weights_bin(model, weights_bin_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar os pesos: {e}")

    else:
        raise FileNotFoundError("Nenhum modelo encontrado nos formatos Keras (`.h5`) ou JSON + binário (`.bin`).")

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
    output_path = 'training/model/model.tflite'
    convert_model(output_path)
