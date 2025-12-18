import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

/*
  Day 10 Goal:
  Multi-class classification using TensorFlow.js
  Classes:
  - Low (< 10)
  - Medium (10–20)
  - High (> 20)
*/

export default function LabelClassifier() {
  // User input number
  const [input, setInput] = useState(null);

  // Training state (used to disable UI while model trains)
  const [training, setTraining] = useState(true);

  // Store prediction probabilities for UI rendering
  const [results, setResults] = useState(null);
  // Store the trained model without triggering re-renders
  const modelRef = useRef(null);

  /*
    Train model ONCE when component mounts
  */
  useEffect(() => {
    async function trainModel() {
      // Create a sequential neural network
      const model = tf.sequential();

      // First hidden layer
      model.add(
        tf.layers.dense({
          units: 16,
          inputShape: [1],
          activation: "relu",
        })
      );

      // Second hidden layer
      model.add(
        tf.layers.dense({
          units: 32,
          activation: "relu",
        })
      );

      // Output layer (3 classes → softmax)
      model.add(
        tf.layers.dense({
          units: 3,
          activation: "softmax",
        })
      );

      // Compile model for multi-class classification
      model.compile({
        loss: "categoricalCrossentropy",
        optimizer: "adam",
        metrics: ["accuracy"],
      });

      /*
        Training data
        Input: numbers
        Labels (one-hot encoded):
        [1,0,0] → Low
        [0,1,0] → Medium
        [0,0,1] → High
      */
      const xs = tf.tensor([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16],
        [17],
        [18],
        [19],
        [20],
        [21],
        [22],
        [23],
        [24],
        [25],
      ]);

      const ys = tf.tensor([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],

        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],

        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
      ]);

      // Train the model
      await model.fit(xs, ys, {
        epochs: 150,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(
              `Epoch ${epoch} — Loss: ${logs.loss}, Accuracy: ${
                logs.acc ?? logs.accuracy
              }`
            );
          },
        },
      });

      // Save trained model
      modelRef.current = model;
      setTraining(false);
    }

    trainModel();
  }, []);

  /*
    Handle classification when user clicks button
  */
  function handleClassify() {
    if (!modelRef.current || input === null) return;

    // Predict probabilities for each class
    const prediction = modelRef.current
      .predict(tf.tensor([[input]]))
      .dataSync();

    const classes = ["Low", "Medium", "High"];

    // Log class probabilities
    prediction.forEach((p, index) => {
      console.log(`${classes[index]}: ${(p * 100).toFixed(2)}%`);
    });

    // Convert predictions into UI-friendly format
    const formattedResults = classes.map((label, index) => ({
      label,
      confidence: (prediction[index] * 100).toFixed(2),
    }));

    setResults(formattedResults);
  }

  /*
    Reset input (optional improvement: retrain model)
  */
  function reset() {
    setInput(null);
    setResults(null);
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold mb-2">
        Day 10 — Multi-Class Classification
      </h1>

      <p className="mb-4">Classify numbers as Low, Medium, or High</p>

      {/* Training indicator */}
      {training && (
        <p className="text-blue-600 italic mb-2">Model Training… please wait</p>
      )}

      {/* User input */}
      <input
        type="number"
        placeholder="Enter a number"
        onChange={(e) => setInput(Number(e.target.value))}
        className="border p-2 my-3"
      />

      {/* Classify button */}
      <button
        onClick={handleClassify}
        disabled={training}
        className="bg-green-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        Classify
      </button>

      {/* Reset */}
      <button onClick={reset} className="border px-4 py-2 mt-3 rounded">
        Reset
      </button>

      {/* ==========================
          Render prediction results
         ========================== */}
      {results && (
        <div className="mt-6 w-full max-w-sm">
          <h2 className="text-lg font-semibold mb-2">Prediction Confidence</h2>

          {results.map((res, index) => (
            <div
              key={index}
              className="flex justify-between p-2 border rounded mb-2"
            >
              <span>{res.label}</span>
              <span className="font-bold">{res.confidence}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
