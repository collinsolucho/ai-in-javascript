import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

export default function Day12Regression() {
  const [input, setInput] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [training, setTraining] = useState(false);
  const [mode, setMode] = useState("normal"); // underfit, normal, overfit
  const [addNoise, setAddNoise] = useState(false);

  const modelRef = useRef(null);

  // ==========================
  // TRAIN MODEL
  // ==========================
  async function trainModel() {
    setTraining(true);
    setPrediction(null);

    const model = tf.sequential();

    // Configure hidden layers based on mode
    if (mode === "underfit") {
      model.add(
        tf.layers.dense({ units: 8, inputShape: [1], activation: "relu" })
      );
    } else if (mode === "normal") {
      model.add(
        tf.layers.dense({ units: 64, inputShape: [1], activation: "relu" })
      );
      model.add(tf.layers.dense({ units: 32, activation: "relu" }));
      model.add(tf.layers.dense({ units: 16, activation: "relu" }));
      model.add(tf.layers.dense({ units: 8, activation: "relu" }));
    } else if (mode === "overfit") {
      model.add(
        tf.layers.dense({ units: 128, inputShape: [1], activation: "relu" })
      );
      model.add(tf.layers.dense({ units: 64, activation: "relu" }));
      model.add(tf.layers.dense({ units: 32, activation: "relu" }));
      model.add(tf.layers.dense({ units: 16, activation: "relu" }));
      model.add(tf.layers.dense({ units: 8, activation: "relu" }));
    }

    model.add(tf.layers.dense({ units: 1 })); // output layer

    model.compile({
      loss: "meanSquaredError",
      optimizer: "adam",
    });

    // ==========================
    // DATASET
    // ==========================
    let xs = tf
      .tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
      .div(10);
    let ys = tf
      .tensor([[1], [4], [9], [16], [25], [36], [49], [64], [81], [100]])
      .div(100);

    if (addNoise) {
      // Add random noise to outputs to see effect on overfitting
      const noise = tf.randomNormal([10, 1], 0, 0.05);
      ys = ys.add(noise);
    }

    // Train
    await model.fit(xs, ys, {
      epochs: 500,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 50 === 0)
            console.log(`Epoch ${epoch} — Loss: ${logs.loss}`);
        },
      },
    });

    modelRef.current = model;
    setTraining(false);
  }

  // ==========================
  // PREDICTION
  // ==========================
  function predictValue() {
    if (!modelRef.current || input === null) return;

    const normInput = input / 10;
    const normOutput = modelRef.current
      .predict(tf.tensor([[normInput]]))
      .dataSync()[0];
    setPrediction((normOutput * 100).toFixed(2));
  }

  function reset() {
    setInput(null);
    setPrediction(null);
    modelRef.current = null;
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold mb-2">
        Day 12 — Interactive Regression
      </h1>
      <p className="mb-4">
        Experiment with underfitting, overfitting, and normalization
      </p>

      {/* Mode selection */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setMode("underfit")}
          className={`px-4 py-2 rounded ${mode === "underfit" ? "bg-blue-600 text-white" : "border"}`}
        >
          Underfit
        </button>
        <button
          onClick={() => setMode("normal")}
          className={`px-4 py-2 rounded ${mode === "normal" ? "bg-blue-600 text-white" : "border"}`}
        >
          Normal
        </button>
        <button
          onClick={() => setMode("overfit")}
          className={`px-4 py-2 rounded ${mode === "overfit" ? "bg-blue-600 text-white" : "border"}`}
        >
          Overfit
        </button>
      </div>

      {/* Noise toggle */}
      <label className="flex items-center gap-2 mb-4">
        <input
          type="checkbox"
          checked={addNoise}
          onChange={(e) => setAddNoise(e.target.checked)}
        />
        Add noise to data
      </label>

      <button
        onClick={trainModel}
        disabled={training}
        className="bg-green-600 text-white px-4 py-2 rounded mb-4 disabled:opacity-50"
      >
        Train Model
      </button>

      <input
        type="number"
        placeholder="Enter a number"
        value={input ?? ""}
        onChange={(e) => setInput(Number(e.target.value))}
        className="border p-2 mb-4"
      />

      <button
        onClick={predictValue}
        disabled={training || !modelRef.current}
        className="bg-blue-600 text-white px-4 py-2 rounded mb-4 disabled:opacity-50"
      >
        Predict
      </button>

      {prediction && (
        <p className="mt-4 text-xl font-bold">Predicted Value: {prediction}</p>
      )}

      <button onClick={reset} className="border px-4 py-2 mt-3 rounded">
        Reset
      </button>

      {training && (
        <p className="text-blue-600 italic mt-2">Training model… please wait</p>
      )}
    </div>
  );
}
