import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';

const API_KEY = "AIzaSyD36C-k9xtxWnkzTv5RxZIf-rqyAtLWed4";
const SPREADSHEET_ID = "1LcUUTuImfcS3inEbR67diqMdlEBvzi9OawPelshhcfE";
const RANGE = "B2:B1000";
const url = `https://sheets.googleapis.com/v4/spreadsheets/${SPREADSHEET_ID}/values/${RANGE}?key=${API_KEY}`;

const requestToSheet = async () => {
  try {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Erro HTTP! Código: ${response.status}`);
    }

    const arquivo = await response.json();

    if (!arquivo.values) {
      throw new Error("Nenhum dado retornado da API.");
    }

    const linhas = arquivo.values
      .map((linha) => {
        if (linha[0] && typeof linha[0] === "string") {
          let valorNumerico = linha[0].replace(",", ".");
          if (!isNaN(valorNumerico)) {
            return parseFloat(valorNumerico);
          } else {
            console.error("Erro: valor inválido detectado:", linha[0]);
            return null;
          }
        } else {
          console.error("Erro: linha vazia ou não numérica:", linha);
          return null;
        }
      })
      .filter((valor) => valor !== null);

    console.log("Valores extraídos:", linhas);

    let dadosEixoX = [];
    let dadosEixoY = [];

    for (let i = 0; i < linhas.length - 1; i++) {
      let valorX = parseFloat(linhas[i + 1]); 
      let valorY = parseFloat(linhas[i]); 

      if (i === linhas.length - 2) {
        valorX = parseFloat(linhas[i]); 
        valorY = parseFloat(linhas[i - 1]);
      }

      if (!isNaN(valorX) && !isNaN(valorY)) {
        dadosEixoX.push(valorX);
        dadosEixoY.push(valorY);
      } else {
        console.error("Erro: valor inválido detectado nos dados de treinamento.");
      }
    }

    // Criação do modelo
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

    const x = tf.tensor(dadosEixoX, [dadosEixoX.length, 1]);
    const y = tf.tensor(dadosEixoY, [dadosEixoY.length, 1]);

    (async () => {
      console.log("Iniciando treinamento...");
      await model.fit(x, y, { epochs: 500 });
      console.log("Treinamento concluído!");

      // Diretório onde o modelo será salvo
      const modelDir = 'training/model';
      if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir, { recursive: true });
      }

      try {
        await model.save(`file://${modelDir}`);
        console.log("Modelo salvo com sucesso no formato SavedModel!");
      } catch (error) {
        console.error("Erro ao salvar o modelo:", error);
      }
    })();
  } catch (error) {
    console.error("Erro ao buscar os dados:", error);
  }
};

requestToSheet();
