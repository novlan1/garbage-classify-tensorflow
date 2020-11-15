import * as tf from "@tensorflow/tfjs";
import { FusedConv2D } from "@tensorflow/tfjs";

const MODEL_DATA_URL = "http://127.0.0.1:8080/model.json";
const CLASSES_DATA_URL = "http://127.0.0.1:8080/classes.json";

let model;

const main = async () => {
  model = await tf.loadLayersModel(MODEL_DATA_URL);
  model.summary();
};



main();

const uploadBtn = document.getElementById("uploadBtn");

uploadBtn.addEventListener("change", onFileChange);

async function onFileChange(e) {
  const file = e.target.files[0];
  console.log(file);

  const classes = await fetch(CLASSES_DATA_URL).then((res) => res.json());

  const img = await file2img(file);

  const pred = await tf.tidy(() => {
    // 输入张量
    const x = img2X(img);
    return model.predict(x);
  });
  pred.print();
  const res = pred
    .arraySync()[0]
    .map((score, index) => {
      return {
        score,
        label: classes[index],
      };
    })
    .sort((a, b) => b.score - a.score);
  console.log(res);
}

function file2img(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (e) => {
      const img = document.createElement("img");
      img.src = e.target.result;
      img.width = 224;
      img.height = 224;
      img.onload = () => resolve(img);
    };
  });
}

function img2X(img) {
  return tf.tidy(() => {
    return tf.browser
      .fromPixels(img)
      .toFloat()
      .sub(255 / 2)
      .div(255 / 2)
      .reshape([1, 224, 224, 3]);
  });
}


// hs output --cors // 启动本地静态服务器命令
// "browserslist": ["last 1 Chrome version"]