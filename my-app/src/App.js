import React from "react";
import { Button, Upload, message, Progress } from "antd";
import "antd/dist/antd.css";
import typeIntroObj from "./intro";
import classesJSON from "./output/classes.json";


import * as tf from "@tensorflow/tfjs";

const MODEL_DATA_URL = 'https://novlan1.github.io/garbage-classify-tensorflow/my-app/src/output/model.json' // 'https://github.com/novlan1/garbage-classify-tensorflow/tree/master/my-app/src/output/model.json'; // 'http://model.uwayfly.com/model.json' // "http://127.0.0.1:8080/model.json";
// const CLASSES_DATA_URL = "http://127.0.0.1:8080/classes.json";

function getBase64(img, callback) {
  const reader = new FileReader();
  reader.addEventListener("load", () => callback(reader.result));
  reader.readAsDataURL(img);
}

function beforeUpload(file) {
  const isJpgOrPng = file.type === "image/jpeg" || file.type === "image/png";

  if (!isJpgOrPng) {
    message.error("You can only upload JPG/PNG file!");
  }

  const isLt2M = file.size / 1024 / 1024 < 2;

  if (!isLt2M) {
    message.error("Image must smaller than 2MB!");
  }

  return isJpgOrPng && isLt2M;
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

class App extends React.PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      predictRes: [],
    };
  }

  async componentDidMount() {
    this.model = await tf.loadLayersModel(MODEL_DATA_URL);
    this.model.summary();
    this.classes = classesJSON; // await fetch(CLASSES_DATA_URL).then((res) => res.json());
  }

  handleChange = (info) => {
    if (info.file.status === "uploading") {
      this.setState({
        loading: true,
        imageUrl: null,
        predictRes: [],
      });
      return;
    }

    // if (info.file.status === "done") {
    getBase64(info.file.originFileObj, (imageUrl) => {
      this.setState({
        imageUrl,
        loading: false,
      });
    });
    // }
    console.log(info);
    this.onPredict(info.file.originFileObj);
  };

  onPredict = async (file) => {
    const img = await file2img(file);

    const pred = await tf.tidy(() => {
      // 输入张量
      const x = img2X(img);
      return this.model.predict(x);
    });

    pred.print();

    const res = pred
      .arraySync()[0]
      .map((score, index) => {
        return {
          score,
          label: this.classes[index],
        };
      })
      .sort((a, b) => b.score - a.score);

    console.log(res);
    this.onHandlePredictResult(res);
  };

  onHandlePredictResult = (data) => {
    const predictObj = [];
    data.map((item) => {
      if (typeIntroObj[item.label]) {
        item.icon = typeIntroObj[item.label]["icon"];
        item.color = typeIntroObj[item.label]["color"];
        item.intro = typeIntroObj[item.label]["info"];
        item.percent = parseInt(item.score * 100);
      }
    });
    this.setState({
      predictRes: data,
    });
    console.log(data);
  };

  render() {
    const { imageUrl, predictRes } = this.state;

    return (
      <div className="wrap">
        <Upload
          // action="https://www.mocky.io/v2/5cc8019d300000980a055e76"
          accept="image/*"
          showUploadList={false}
          beforeUpload={beforeUpload}
          onChange={this.handleChange}
          className="upload"
        >
          <Button
            type="primary"
            style={{ width: "calc(100vw - 5rem)", marginTop: "1rem" }}
          >
            上传
          </Button>
        </Upload>

        <div>
          {imageUrl && (
            <img src={imageUrl} style={{ width: "100%", marginTop: "1rem" }} />
          )}
        </div>

        <div style={{ marginTop: "1rem" }}>
          <div>识别结果：</div>
          {!!predictRes.length && (
            <div style={{ display: "flex" }}>
              <span>
                <img className="garbage-type" src={predictRes[0].icon} />
              </span>
              <span
                style={{ marginLeft: "0.2rem", color: predictRes[0].color }}
              >
                <h4 style={{ color: predictRes[0].color }}>
                  {predictRes[0].label}
                </h4>
                <span>{predictRes[0].intro}</span>
              </span>
            </div>
          )}
        </div>

        <div style={{ margin: "1rem 0" }}>
          <div className="type-row">
            <span className="type-row-left">类别</span>
            <span className="type-row-right">匹配度</span>
          </div>
          {!!predictRes.length &&
            predictRes.map((predictItem) => (
              <div className="type-row" key={predictItem.label}>
                <span className="type-row-left">{predictItem.label}</span>
                <span className="type-row-right">
                  <Progress percent={predictItem.percent} />
                </span>
              </div>
            ))}
        </div>
      </div>
    );
  }
}

export default App;

