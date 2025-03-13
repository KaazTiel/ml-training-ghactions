import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import numpy as np
import os
import tensorflowjs as tfjs  # Importando o tensorflowjs

def load_weights_h5(model, weights_h5_path):
    """Carrega pesos do modelo a partir de um arquivo .h5."""
    if not os.path.exists(weights_h5_path):
        raise FileNotFoundError(f"Arquivo de pesos '{weights_h5_path}' não encontrado.")

    try:
        model.load_weights(weights_h5_path)
        print("Pesos do modelo carregados com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar os pesos do .h5: {e}")

def convert_model(output_path_tflite, output_path_tfjs):
    """Converte um modelo salvo em diferentes formatos (Keras `.h5`, JSON+pesos) para TFLite e TensorFlow.js."""
    
    # Opções de arquivos possíveis
    model_json_path = 'training/model/model.json'
    weights_bin_path = 'training/model/weights.bin'
    weights_h5_path = 'training/model/model.weights.h5'
    keras_model_path = 'training/model/model.h5'

    model = None

    # 1️⃣ Verifica se há um modelo Keras completo salvo como `.h5`
    if os.path.exists(keras_model_path):
        print(f"Modelo encontrado em formato Keras: {keras_model_path}")
        try:
            model = load_model(keras_model_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o modelo Keras: {e}")

    # 2️⃣ Verifica se há um modelo salvo como JSON + pesos `.h5`
    elif os.path.exists(model_json_path) and os.path.exists(weights_h5_path):
        print(f"Modelo encontrado no formato JSON + pesos .h5: {model_json_path}, {weights_h5_path}")
        
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
            load_weights_h5(model, weights_h5_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar os pesos .h5: {e}")

    # 3️⃣ Verifica se há um modelo salvo como JSON + pesos `.bin` (alternativo)
    elif os.path.exists(model_json_path) and os.path.exists(weights_bin_path):
        print(f"Modelo encontrado no formato JSON + pesos binários: {model_json_path}, {weights_bin_path}")
        
        # Carrega a arquitetura do modelo
        try:
            with open(model_json_path, 'r') as json_file:
                model_json = json_file.read().strip()
            if not model_json:
                raise ValueError("O arquivo model.json está vazio ou inválido.")

            model = model_from_json(model_json)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o modelo a partir do JSON: {e}")

        # ⚠️ Aqui precisaria implementar corretamente a carga dos pesos binários
        raise NotImplementedError("Carregamento de pesos .bin ainda não suportado corretamente.")

    else:
        raise FileNotFoundError("Nenhum modelo encontrado nos formatos esperados (`.h5`, JSON + `.h5`, JSON + `.bin`).")

    # Converte o modelo para TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
    except Exception as e:
        raise RuntimeError(f"Erro na conversão para TFLite: {e}")

    # Salva o modelo convertido em TFLite
    with open(output_path_tflite, 'wb') as f:
        f.write(tflite_model)

    print(f"Modelo convertido para TFLite e salvo em: {output_path_tflite}")

    # Converte o modelo para TensorFlow.js
    try:
        tfjs.converters.save_keras_model(model, output_path_tfjs)
    except Exception as e:
        raise RuntimeError(f"Erro na conversão para TensorFlow.js: {e}")

    print(f"Modelo convertido para TensorFlow.js e salvo em: {output_path_tfjs}")

if __name__ == "__main__":
    output_path_tflite = 'training/model/model.tflite'
    output_path_tfjs = 'training/model/model_tfjs'
    convert_model(output_path_tflite, output_path_tfjs)
