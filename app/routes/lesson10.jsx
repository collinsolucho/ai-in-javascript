import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

export default function RegressionDemo() {
  const [input, setInput] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [training, setTraining] = useState(true);

  const modelRef = useRef(null);

  // ==========================
  // Train model once on mount
  // ==========================
  useEffect(() => {
    async function trainModel() {
      const model = tf.sequential();
      //the hidden layer helps to learn curves
      //they add capacicity and reduce underfitting
      //helps to learn curves eg y=x^2 unlike lesson 5 where it only learns one curve input eg y=mx+c
      model.add(
        tf.layers.dense({
          units: 64, // More "segments" to build the curve
          inputShape: [1],
          activation: "relu", //adds non-linearity Without it, even many layers collapse back into a straight line
        })
      );

      model.add(
        tf.layers.dense({
          units: 32, // A second layer to refine the shape
          activation: "relu",
        })
      );

      model.add(
        tf.layers.dense({
          units: 16, // A third layer to refine the shape
          activation: "relu",
        })
      );

      model.add(
        tf.layers.dense({
          units: 8, // A forth layer to refine the shape
          activation: "relu",
        })
      );

      model.add(
        tf.layers.dense({
          units: 1, // regression output
        })
      );
      //Architecture decides what can be learned.
      // Optimizer decides how fast it is learned.
      model.compile({
        loss: "meanSquaredError",
        optimizer: "adam", //adaptive & production ready
      });
      // without input,output normalization
      //normalization reduces loss by a great deal and increase output value

      //   const xs = tf.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);

      //   const ys = tf.tensor([
      //     [1],
      //     [4],
      //     [9],
      //     [16],
      //     [25],
      //     [36],
      //     [49],
      //     [64],
      //     [81],
      //     [100],
      //   ]);
      //with input/output normalization
      const xs = tf
        .tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        .div(10);
      const ys = tf
        .tensor([[1], [4], [9], [16], [25], [36], [49], [64], [81], [100]])
        .div(100);

      await model.fit(xs, ys, {
        epochs: 1000,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch} — Loss: ${logs.loss}`);
          },
        },
      });

      modelRef.current = model;
      setTraining(false);
    }

    trainModel();
  }, []);

  // ==========================
  // Predict numeric output
  // ==========================
  function predictValue() {
    if (!modelRef.current || input === null) return;
    // without normalization
    // const output = modelRef.current.predict(tf.tensor([[input]])).dataSync()[0];
    // with normalization of prediction
    // Observation: Training stabilizes, predictions are accurate and generalize better.

    let normInput = input / 10;
    let denorm = modelRef.current.predict(tf.tensor([normInput])).dataSync()[0];
    let output = denorm * 100;
    setPrediction(output.toFixed(2));
  }

  function reset() {
    setInput(null);
    setPrediction(null);
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold mb-2">Day 11 — Regression</h1>

      <p className="mb-4">Predict output using a trained regression model</p>

      {training && (
        <p className="text-blue-600 italic mb-2">Training model… please wait</p>
      )}

      <input
        type="number"
        placeholder="Enter a number"
        onChange={(e) => setInput(Number(e.target.value))}
        className="border p-2 my-3"
      />

      <button
        onClick={predictValue}
        disabled={training}
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        Predict
      </button>

      {prediction && (
        <p className="mt-4 text-xl font-bold">Predicted Value: {prediction}</p>
      )}

      <button onClick={reset} className="border px-4 py-2 mt-3 rounded">
        Reset
      </button>
    </div>
  );
}
