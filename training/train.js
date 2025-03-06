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

    // Garantir que o diretório exista antes de salvar
    const modelDir = 'training/model';
    if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, { recursive: true });  // Cria o diretório se não existir
    }

    // Salvar a arquitetura do modelo (JSON)
    try {
        const modelJson = model.toJSON(); // Obtém a arquitetura em JSON
        fs.writeFileSync(path.join(modelDir, 'model.json'), JSON.stringify(modelJson));
        console.log("Modelo (JSON) salvo com sucesso!");

        // Salvar os pesos do modelo (binário)
        const weights = model.getWeights(); // Obtém os pesos do modelo
        const weightBuffer = Buffer.concat(weights.map(weight => Buffer.from(weight.dataSync())));
        fs.writeFileSync(path.join(modelDir, 'weights.bin'), weightBuffer);
        console.log("Pesos (weights.bin) salvos com sucesso!");
    } catch (error) {
        console.error("Erro ao salvar o modelo:", error);
    }
})();
