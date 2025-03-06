import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os

def convert_model(model_json_path, weights_bin_path, output_path):
    # Verificar se os arquivos de modelo e pesos existem
    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Arquivo de arquitetura do modelo não encontrado: {model_json_path}")
    if not os.path.exists(weights_bin_path):
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_bin_path}")

    try:
        # Carregar a arquitetura do modelo a partir do arquivo JSON
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)

        # Carregar os pesos a partir do arquivo binário
        model.load_weights(weights_bin_path)

        # Converter o modelo para o formato TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Salvar o modelo convertido
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Modelo convertido com sucesso para {output_path}")

    except Exception as e:
        print(f"Ocorreu um erro durante a conversão do modelo: {e}")
        raise

if __name__ == "__main__":
    model_json_path = 'training/model/model.json'
    weights_bin_path = 'training/model/weights.bin'
    output_path = 'training_model.tflite'
    convert_model(model_json_path, weights_bin_path, output_path)