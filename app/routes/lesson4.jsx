import { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export default function SimpleNN() {
  useEffect(() => {
    // Run only once on mount (because dependency array is empty)
    async function run() {
      // ===============================
      // 1. CREATE MODEL (Sequential)
      // ===============================
      // A sequential model is a straight-line stack of layers.
      const model = tf.sequential();

      // Add a Dense (fully connected) layer
      // units: number of neurons
      // inputShape: the model expects 1 number as input
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

      // ===============================
      // 2. COMPILE MODEL
      // ===============================
      // loss: how wrong the model is
      // optimizer: how the model learns (sgd = stochastic gradient descent)
      model.compile({
        loss: "meanSquaredError",
        optimizer: "sgd",
      });

      // ===============================
      // 3. TRAINING DATA
      // ===============================
      // xs → inputs
      // ys → targets/outputs
      // Pattern: y = 2x + 1
      const xs = tf.tensor([1, 2, 3, 4, 5, 6]);
      const ys = tf.tensor([3, 5, 7, 9, 11, 13]);

      // ===============================
      // 4. TRAIN THE MODEL
      // ===============================
      console.log("Training...");
      await model.fit(xs, ys, {
        epochs: 20, // number of training cycles
        callbacks: {
          // tracks progress every epoch
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch} — Loss: ${logs.loss}`);
          },
        },
      });

      // ===============================
      // 5. MAKE A PREDICTION
      // ===============================
      // The model should learn that y ≈ 2x + 1
      const prediction = model.predict(tf.tensor([5]));

      console.log("Prediction Tensor:", prediction);
      // print() logs the value inside the tensor
      prediction.print();
    }

    // Initialize the model training
    run();
  }, []); // Empty array = run once only

  // ===============================
  // UI
  // ===============================
  return (
    <div className="min-h-screen flex flex-col items-center justify-center  p-6">
      <h1 className="text-3xl font-bold mb-6">Day 8 — First Neural Network</h1>
      <p className="text-lg font-bold mb-6">
        Open your console to see training logs and prediction output.
      </p>
    </div>
  );
}
