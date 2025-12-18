import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

export default function MultiClassClassifier() {
  const [input, setInput] = useState(null);
  const [label, setLabel] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [training, setTraining] = useState(true);

  const modelRef = useRef(null);

  useEffect(() => {
    async function train() {
      const model = tf.sequential();

      model.add(
        tf.layers.dense({ units: 8, inputShape: [1], activation: "relu" })
      );
      model.add(tf.layers.dense({ units: 3, activation: "softmax" }));
      //3 units define the 3 neurons 4 small,medium,large
      //softmax takes output from neurons & converts 2 probality distribution
      model.compile({
        loss: "categoricalCrossentropy",
        optimizer: "adam",
        metrics: ["accuracy"],
      });

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
      ]);
      // : Ensure your xs and ys tensors have the exact same number of elements (the outer dimension)
      const ys = tf.tensor([
        // this called one-hot encoding. instead of 0,1 use a vector with length equal 2 the no of classes (3)
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
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
      ]);

      await model.fit(xs, ys, { epochs: 60 });
      modelRef.current = model;
      setTraining(false);
    }

    train();
  }, []);

  function classify() {
    if (!modelRef.current || input === null) return;

    const prediction = modelRef.current
      .predict(tf.tensor([[input]]))
      .dataSync();

    const classes = ["Small", "Medium", "Large"];
    //The model outputs three probabilities (e.g., [0.05, 0.85, 0.10]).
    // Math.max(...prediction) finds the highest probability (e.g., 0.85).
    // maxIndex finds the index of that highest probability (e.g., index 1).
    const maxIndex = prediction.indexOf(Math.max(...prediction));

    setLabel(classes[maxIndex]);
    // The index is mapped to the final class label using the classes array (classes[1] is "Medium").
    // The confidence is simply that maximum probability.
    setConfidence((prediction[maxIndex] * 100).toFixed(2));
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold">
        Day 10 — Multi-Class Classification
      </h1>
      <p className="mb-3">Classify a number as Small, Medium or Large</p>

      {training && <p className="text-blue-600">Training model…</p>}

      <input
        type="number"
        onChange={(e) => setInput(Number(e.target.value))}
        className="border p-2 my-3"
      />

      <button
        onClick={classify}
        disabled={training}
        className="bg-green-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        Classify
      </button>

      {label && (
        <p className="mt-4 text-xl font-bold">
          Result: {label} ({confidence}%)
        </p>
      )}
    </div>
  );
}
