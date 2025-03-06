import tensorflow as tf
from tensorflow.keras.models import model_from_json

def convert_model(model_json_path, weights_bin_path, output_path):
    # Carregar a arquitetura do modelo a partir do arquivo JSON
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()

    # Garantir que o modelo seja carregado a partir do JSON
    model = model_from_json(model_json)

    # Carregar os pesos a partir do arquivo binário
    try:
        model.load_weights(weights_bin_path)
    except Exception as e:
        print(f"Erro ao carregar os pesos: {e}")
        exit(1)

    # Converter o modelo para o formato TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Salvar o modelo convertido
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    model_json_path = 'training/model/model.json'
    weights_bin_path = 'training/model/weights.bin'
    output_path = 'training_model.tflite'
    convert_model(model_json_path, weights_bin_path, output_path)
