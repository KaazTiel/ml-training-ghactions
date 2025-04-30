const tf = require("@tensorflow/tfjs-node"); // Ainda usa a versão node para treinar com performance
const ARIMA = require("arima");

// ====== Configurações da Google Sheets API ======
const API_KEY = "AIzaSyD36C-k9xtxWnkzTv5RxZIf-rqyAtLWed4";
const SPREADSHEET_ID = "1cTcdqJtk1qfWeCmfOAKkLae2vZfEd8rYLWzyQWPTb4A";
const RANGE = "B3:B1000"; // Coluna com os valores
const url = `https://sheets.googleapis.com/v4/spreadsheets/${SPREADSHEET_ID}/values/${RANGE}?key=${API_KEY}`;

// ===== Funções de normalização =====
const normalize = (x, min, max) => (x - min) / (max - min);
const denormalize = (x, min, max) => x * (max - min) + min;

// ===== Função principal =====
const trainModel = async () => {
  // Busca os dados da Google Sheets
  const response = await fetch(url);
  const data = await response.json();

  if (!data.values || data.values.length === 0) {
    console.error("Nenhum dado encontrado na planilha.");
    return;
  }

  // Converte os dados em uma série temporal de floats
  const ts = data.values
    .map((linha) => {
      const val = linha[0]?.replace(",", ".");
      const num = parseFloat(val);
      return isNaN(num) ? null : num;
    })
    .filter((val) => val !== null)
    .reverse(); // Ordena do mais antigo para o mais recente

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
  const fs = require("fs");
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  await model.save(`file://${dir}`);
  console.log("✅ Modelo salvo com sucesso!");

  // Previsão final híbrida
  const inputLSTM = tf.tensor3d([normalized.slice(-window).map((v) => [v])]);
  const predLSTM = model.predict(inputLSTM);
  const predValue = predLSTM.dataSync()[0];

  console.log("📈 Previsão final (SARIMA + LSTM):", denormalize(predValue, min, max).toFixed(2));
};

trainModel();
