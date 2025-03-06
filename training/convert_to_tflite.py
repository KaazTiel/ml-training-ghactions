import tensorflow as tf
import os

def convert_model(model_path, output_path):
    # Carregar o modelo Keras
    model = tf.keras.models.load_model(model_path)
    
    # Converter para o formato TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Salvar o modelo convertido
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Modelo convertido e salvo em {output_path}')

if __name__ == '__main__':
    model_path = 'training/model/model.h5'  # Caminho do modelo Keras
    output_path = 'training_model.tflite'  # Caminho do modelo TFLite
    convert_model(model_path, output_path)
