const fs = require("fs");
const parse = require("csv-parse/sync");
const ARIMA = require("arima");

const file = fs.readFileSync("training/Cotacoes_Filtradas_nov_abril.csv");
const records = parse.parse(file, {
  columns: true,
  skip_empty_lines: true,
  delimiter: ",", 
});

const ts = records.map((r) => parseFloat(r["Último"])).reverse(); 

const arima = new ARIMA({
  p: 3,
  d: 1,
  q: 3,
  P: 1,
  D: 1,
  Q: 0,
  s: 5,
  verbose: false,
});

arima.train(ts);

const modelDir = "training/model";
if (!fs.existsSync(modelDir)) {
  fs.mkdirSync(modelDir, { recursive: true });
}

fs.writeFileSync(`${modelDir}/sarima-model.json`, JSON.stringify(arima.toJSON()));

console.log("✅ Modelo SARIMA salvo com sucesso.");
