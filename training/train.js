const tf = require("@tensorflow/tfjs-node");
const ARIMA = require("arima");
const fs = require("fs");
const fetch = require("node-fetch"); // caso não tenha fetch global

// ====== Configurações da Google Sheets API ======
const API_KEY = "AIzaSyD36C-k9xtxWnkzTv5RxZIf-rqyAtLWed4";
const SPREADSHEET_ID = "1cTcdqJtk1qfWeCmfOAKkLae2vZfEd8rYLWzyQWPTb4";
const RANGE = "A3:B1000"; // Data em A, Preço em B
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

  // Extrai pares {date, price} e filtra valores válidos
  const datePricePairs = data.values
    .map(([dateStr, priceStr]) => {
      const price = parseFloat(priceStr?.replace(",", "."));
      return !isNaN(price) ? { date: dateStr, price } : null;
    })
    .filter((entry) => entry !== null)
    .reverse(); // ordem do mais antigo para o mais recente

  // Apenas os preços para o modelo
  const ts = datePricePairs.map(({ price }) => price);

  // SARIMA
  const arima = new ARIMA({ p: 3, d: 1, q: 3, P: 1, D: 1, Q: 0, s: 5 }).train(ts);
  const [sarimaPred] = arima.predict(1);
  const predValueSARIMA = sarimaPred[0];

  // Estende a série com a predição SARIMA
  const extended = [...ts, predValueSARIMA];

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

  // Previsão final híbrida LSTM com input da última janela
  const inputLSTM = tf.tensor3d([normalized.slice(-window).map((v) => [v])]);
  const predLSTM = model.predict(inputLSTM);
  const predValueLSTM = predLSTM.dataSync()[0];

  // Previsão final combinada (pode ajustar aqui, por ex. média simples)
  // Aqui só uso LSTM para saída, mas poderia combinar SARIMA+LSTM se quiser
  const finalPrediction = denormalize(predValueLSTM, min, max);

  console.log("📈 Previsão final (SARIMA + LSTM):", finalPrediction.toFixed(2));

  // Salvar JSON com a predição no topo
  const jsonData = {
    prediction: finalPrediction.toFixed(2),
    data: datePricePairs,
  };

  fs.writeFileSync("training/prediction_data.json", JSON.stringify(jsonData, null, 2));
  console.log("✅ JSON com datas e preços salvo com predição!");
};

trainModel();
