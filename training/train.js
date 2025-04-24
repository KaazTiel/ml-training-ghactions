import fs from "fs";
import { parse } from "csv-parse/sync";
import ARIMA from "arima";

// leitura do arquivo
const file = fs.readFileSync("./Cotacoes_Filtradas_nov_abril.csv");
const records = parse(file, {
  columns: true,
  skip_empty_lines: true,
  delimiter: ",",
});

// extração da série temporal
const ts = records.map((r) => parseFloat(r["Último"])).reverse();

// criação e treino do modelo
const arima = new ARIMA({
  p: 3,
  d: 1,
  q: 3,
  P: 1,
  D: 1,
  Q: 0,
  s: 5,
  verbose: false,
}).train(ts);

// criação da pasta se necessário
const modelDir = "training/model";
if (!fs.existsSync(modelDir)) {
  fs.mkdirSync(modelDir, { recursive: true });
}

// salvamento do modelo treinado
fs.writeFileSync(`${modelDir}/sarima-model.json`, JSON.stringify(arima.toJSON()));

console.log("Modelo SARIMA salvo em JSON.");
