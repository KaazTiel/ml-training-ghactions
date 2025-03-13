import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
import tensorflowjs as tfjs
import numpy as np
import os
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    """
    Converte modelos salvos nos formatos Keras, JSON + pesos ou TensorFlow.js para TFLite e TensorFlow.js.

    Args:
        output_path_tflite: Caminho onde o modelo convertido para TFLite será salvo.
        output_path_tfjs: Caminho onde o modelo convertido para TensorFlow.js será salvo.
    """
    def __init__(self, output_path_tflite, output_path_tfjs):
        self.output_path_tflite = output_path_tflite
        self.output_path_tfjs = output_path_tfjs

        # Caminhos para os modelos salvos
        self.model_json_path = 'training/model/model.json'
        self.weights_bin_path = 'training/model/weights.bin'
        self.keras_model_path = 'training/model/model.h5'

        # Parâmetros para MobileNet
        self.input_node_name = 'the_input'
        self.image_size = 224
        self.alpha = 0.25
        self.depth_multiplier = 1
        self._input_shape = (1, self.image_size, self.image_size, 3)
        self.depthwise_conv_layer = 'conv_pw_13_relu'

    def convert(self):
        model = self.load_model()
        if model:
            self.save_keras_model(model)
            self.convert_to_tflite(model)
            self.convert_to_tfjs(model)

    def load_model(self):
        """Carrega o modelo conforme os arquivos disponíveis."""
        model = None

        # 1️⃣ Verifica se há um modelo Keras completo salvo como `.h5`
        if os.path.exists(self.keras_model_path):
            logger.info(f"Modelo encontrado em formato Keras: {self.keras_model_path}")
            try:
                model = load_model(self.keras_model_path)
            except Exception as e:
                logger.error(f"Erro ao carregar o modelo Keras: {e}")

        # 2️⃣ Verifica se há um modelo salvo como JSON + pesos `.bin`
        elif os.path.exists(self.model_json_path) and os.path.exists(self.weights_bin_path):
            logger.info(f"Modelo encontrado no formato JSON + pesos binários: {self.model_json_path}, {self.weights_bin_path}")
            try:
                with open(self.model_json_path, 'r') as json_file:
                    model_json = json_file.read().strip()
                if not model_json:
                    raise ValueError("O arquivo model.json está vazio ou inválido.")
                model = model_from_json(model_json)
            except Exception as e:
                logger.error(f"Erro ao carregar o modelo a partir do JSON: {e}")
                return None

            try:
                self.load_weights_bin(model)
                logger.info("Pesos carregados com sucesso a partir do arquivo .bin.")
            except Exception as e:
                logger.error(f"Erro ao carregar os pesos .bin: {e}")
                return None
        else:
            logger.error("Nenhum modelo encontrado nos formatos esperados (`.h5`, JSON + `.bin`).")
            return None
        
        return model

    def load_weights_bin(self, model):
        """Carrega os pesos a partir de um arquivo binário (.bin)"""
        logger.info("Carregando pesos do arquivo .bin...")
        try:
            # Lê o arquivo binário de pesos
            weights = np.fromfile(self.weights_bin_path, dtype=np.float32)
            logger.info(f"Pesos lidos do arquivo .bin com {len(weights)} elementos.")

            # Atribui os pesos ao modelo
            # Assumindo que o número de pesos corresponde ao número de camadas do modelo
            # O TensorFlow espera que os pesos sejam atribuídos na ordem correta, camada por camada
            weight_shapes = [layer.get_weights()[0].shape for layer in model.layers if len(layer.get_weights()) > 0]
            current_idx = 0
            for i, layer in enumerate(model.layers):
                if len(layer.get_weights()) > 0:
                    weight_shape = weight_shapes[i]
                    weight_size = np.prod(weight_shape)  # Tamanho do peso dessa camada
                    # Aqui ajustamos a atribuição dos pesos para cada camada corretamente
                    layer.set_weights([weights[current_idx:current_idx + weight_size].reshape(weight_shape)])
                    current_idx += weight_size
            logger.info("Pesos binários atribuídos ao modelo com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar pesos binários: {e}")
            raise e

    def save_keras_model(self, model):
        """Salva o modelo Keras mesclado com a arquitetura MobileNet."""
        base_model = self.get_base_model()
        self.merged_model = self.merge(base_model, model)
        logger.info("Modelo Keras mesclado salvo.")

    def get_base_model(self):
        """Constrói o modelo base MobileNet."""
        input_tensor = tf.keras.Input(shape=self._input_shape[1:], name=self.input_node_name)
        base_model = tf.keras.applications.MobileNet(input_shape=self._input_shape[1:],
                                                     alpha=self.alpha,
                                                     depth_multiplier=self.depth_multiplier,
                                                     input_tensor=input_tensor,
                                                     include_top=False)
        return base_model

    def merge(self, base_model, top_model):
        """Mescla o modelo base com a camada de classificação."""
        logger.info("Mesclando modelo base com o modelo de classificação...")
        layer = base_model.get_layer(self.depthwise_conv_layer)
        model = tf.keras.Model(inputs=base_model.input, outputs=top_model(layer.output))
        return model

    def convert_to_tflite(self, model):
        """Converte o modelo para o formato TFLite."""
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            with open(self.output_path_tflite, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"Modelo convertido para TFLite e salvo em: {self.output_path_tflite}")
        except Exception as e:
            logger.error(f"Erro na conversão para TFLite: {e}")

    def convert_to_tfjs(self, model):
        """Converte o modelo para o formato TensorFlow.js."""
        try:
            tfjs.converters.save_keras_model(model, self.output_path_tfjs)
            logger.info(f"Modelo convertido para TensorFlow.js e salvo em: {self.output_path_tfjs}")
        except Exception as e:
            logger.error(f"Erro na conversão para TensorFlow.js: {e}")


if __name__ == "__main__":
    output_path_tflite = 'training/model/model.tflite'
    output_path_tfjs = 'training/model/model_tfjs'
    converter = ModelConverter(output_path_tflite, output_path_tfjs)
    converter.convert()
