const tf = require("@tensorflow/tfjs-node");
const ARIMA = require("arima");
const fs = require("fs");
const fetch = require("node-fetch");

// ====== Configurações da Google Sheets API ======
const API_KEY = "AIzaSyD36C-k9xtxWnkzTv5RxZIf-rqyAtLWed4";
const SPREADSHEET_ID = "1cTcdqJtk1qfWeCmfOAKkLae2vZfEd8rYLWzyQWPTb4A";
const RANGE = "A3:B1000"; // Data na coluna A, Preço na coluna B
const url = `https://sheets.googleapis.com/v4/spreadsheets/${SPREADSHEET_ID}/values/${RANGE}?key=${API_KEY}`;

// ===== Funções de normalização =====
const normalize = (x, min, max) => (x - min) / (max - min);
const denormalize = (x, min, max) => x * (max - min) + min;

const trainModel = async () => {
  // Busca os dados da planilha
  const response = await fetch(url);
  const data = await response.json();

  if (!data.values || data.values.length === 0) {
    console.error("Nenhum dado encontrado na planilha.");
    return;
  }

  // Extrai datas e preços, filtra entradas inválidas
  const datePricePairs = data.values
    .map(([dateStr, priceStr]) => {
      const price = parseFloat(priceStr?.replace(",", "."));
      return !isNaN(price) && dateStr ? { date: dateStr, price } : null;
    })
    .filter((entry) => entry !== null)
    .reverse(); // ordena do mais antigo para o mais recente

  // Separar arrays para datas e preços
  const dates = datePricePairs.map((d) => d.date);
  const prices = datePricePairs.map((d) => d.price);

  // Treinar modelo SARIMA e prever
  const arima = new ARIMA({ p: 3, d: 1, q: 3, P: 1, D: 1, Q: 0, s: 5 }).train(prices);
  const [sarimaPred] = arima.predict(1);
  const extendedPrices = [...prices, sarimaPred[0]];

  // Normalização
  const min = Math.min(...extendedPrices);
  const max = Math.max(...extendedPrices);
  const normalized = extendedPrices.map((x) => normalize(x, min, max));

  // Preparar dados para LSTM
  const window = 5;
  const xs = [], ys = [];
  for (let i = 0; i < normalized.length - window; i++) {
    xs.push(normalized.slice(i, i + window).map((v) => [v]));
    ys.push([normalized[i + window]]);
  }

  const xsTensor = tf.tensor3d(xs);
  const ysTensor = tf.tensor2d(ys);

  // Construir e treinar o modelo LSTM
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

  // Previsão final LSTM (com entrada da última janela)
  const inputLSTM = tf.tensor3d([normalized.slice(-window).map((v) => [v])]);
  const predLSTM = model.predict(inputLSTM);
  const predValueLSTM = predLSTM.dataSync()[0];
  const finalPrediction = denormalize(predValueLSTM, min, max);

  console.log("📈 Previsão final (SARIMA + LSTM):", finalPrediction.toFixed(2));

  // Salvar JSON com dados originais + predição (com a previsão como um novo objeto)
  const jsonData = {
    prediction: finalPrediction.toFixed(2),
    data: [...datePricePairs, { date: "Previsão", price: finalPrediction }],
  };

  fs.writeFileSync("training/model/prediction_data.json", JSON.stringify(jsonData, null, 2));
  console.log("✅ JSON com datas, preços e predição salvo!");
};

trainModel();
