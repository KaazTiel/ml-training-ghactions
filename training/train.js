import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';

// Criando um conjunto de dados de exemplo (regressão linear)
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]); // Entradas
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]); // Saídas esperadas

// Criando um modelo de rede neural simples
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

(async () => {
    console.log("Iniciando treinamento...");
    await model.fit(xs, ys, { epochs: 100 });
    console.log("Treinamento concluído!");

    // Diretório onde o modelo será salvo
    const modelDir = 'training/model';
    if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, { recursive: true });  // Cria o diretório se não existir
    }

    try {
        // Salvar o modelo no formato Keras (.h5)
        await model.save(`file://${modelDir}`);

        console.log("Modelo salvo com sucesso no formato Keras (.h5)!");
    } catch (error) {
        console.error("Erro ao salvar o modelo:", error);
    }
})();
