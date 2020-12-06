const getData = require("./loadData");
const tf = require("@tensorflow/tfjs-node");

const TRAIN_DIR = "garbage_images/train";
const OUTPUT_DIR = "app/output";
const MOBILE_NET_URL =
  "https://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/mobilenet/web_model/model.json";

/**
 * 定义模型
 * 迁移训练：截断模型 + 双层神经网络
 * 迁移训练指复用已经训练好的模型
 */

const defineModel = async (classes) => {
  /**
   * 步骤:
   * 1. 加载预训练模型MobileNet并截断（MobileNet 参数相对较少，可以跑在移动端）
   * 2. 连接自己定义的双层神经网络
   */
  const mobileNet = await tf.loadLayersModel(MOBILE_NET_URL);
  // 打印模型结构
  mobileNet.summary();
  // 查看在哪层截断
  // mobileNet.layers.map((layer, index) => console.log([layer.name, index]))

  // 定义自己的模型
  const model = tf.sequential();
  for (let i = 0; i <= 86; i++) {
    const layer = mobileNet.layers[i];
    // 设置这些层不参与训练
    layer.trainable = false;
    model.add(layer);
  }

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 10, //神经元个数，超参数
      activation: "relu", // 激活函数，解决非线性问题
    })
  );

  model.add(
    tf.layers.dense({
      units: classes.length, // 不是超参数，就是类的个数
      activation: "softmax", // 多分类
    })
  );

  // 打印自己的迁移模型结构
  model.summary();
  return model;
};

/**
 * 训练模型
 * 1. 定义损失函数和优化器
 * 2. 使用tensorflow.js的fit方法进行训练（拟合）
 * 3. 使用tensorflow.js的save方法保存模型到文件
 */
const trainModel = async (model, ds) => {
  model.compile({
    loss: "sparseCategoricalCrossentropy",
    optimizer: tf.train.adam(),
    metrics: ["acc"], // 度量单位
  });

  // await model.fit(xs, ys, {
  //   epochs: 10, // 几轮训练
  // });

  await model.fitDataset(ds, {
    epochs: 10,
  });

  await model.save(`file://${process.cwd()}/${OUTPUT_DIR}/`);
};

const main = async () => {
  // 一、加载数据
  const { ds, classes } = await getData(TRAIN_DIR, OUTPUT_DIR);
  // console.log(ds, classes);

  // 二、定义模型
  const model = await defineModel(classes);

  // 三、训练模型
  await trainModel(model, ds);
};

main();
