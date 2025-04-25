import * as tf from "@tensorflow/tfjs-node";
import { ARIMA } from "arima";
import fs from "fs";
import parse from "csv-parse/sync";

// ===== Configurações =====
const FILE_PATH = "training/Cotacoes_Filtradas_nov_abril.csv";
const COLUNA = "Último"; // Nome da coluna com os preços

// ===== Funções de normalização =====
const normalize = (x, min, max) => (x - min) / (max - min);
const denormalize = (x, min, max) => x * (max - min) + min;

// ===== Função principal =====
const trainModel = async () => {
  // Lê e parseia o CSV
  const file = fs.readFileSync(FILE_PATH);
  const records = parse.parse(file, {
    columns: true,
    skip_empty_lines: true,
    delimiter: ",",
  });

  // Extrai os dados da coluna desejada e ordena do mais antigo ao mais recente
  const ts = records.map((r) => parseFloat(r[COLUNA].replace(",", "."))).reverse();

  // SARIMA
  const arima = new ARIMA({ p: 3, d: 1, q: 3, P: 1, D: 1, Q: 0, s: 5 }).train(ts);
  const [sarimaPred] = arima.predict(1);
  const extended = [...ts, sarimaPred[0]];

  // Normalização
  const min = Math.min(...extended);
  const max = Math.max(...extended);
  const normalized = extended.map((x) => normalize(x, min, max));

  // Preparar janelas para LSTM
  const window = 5;
  const xs = [], ys = [];
  for (let i = 0; i < normalized.length - window; i++) {
    xs.push(normalized.slice(i, i + window).map((v) => [v]));
    ys.push([normalized[i + window]]);
  }

  const xsTensor = tf.tensor3d(xs);
  const ysTensor = tf.tensor2d(ys);

  // Modelo LSTM
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units: 50, inputShape: [window, 1] }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ loss: "meanSquaredError", optimizer: "adam" });

  console.log("⏳ Treinando modelo...");
  await model.fit(xsTensor, ysTensor, { epochs: 100, batchSize: 8 });
  console.log("✅ Modelo treinado!");

  // Salvar modelo
  const dir = "training/model";
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  await model.save(`file://${dir}`);
  console.log("✅ Modelo salvo com sucesso!");
};

trainModel();
