import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function NeuralInput() {
  // Stores the number the user typed
  const [userNumber, setUserNumber] = useState(null);

  // Stores the model prediction result
  const [prediction, setPrediction] = useState(null);

  // useRef holds the trained model so it persists across re-renders
  const modelRef = useRef(null);

  // =======================================
  // TRAIN MODEL ONCE WHEN COMPONENT MOUNTS
  // =======================================
  useEffect(() => {
    async function run() {
      // 1. Create a simple sequential model
      const model = tf.sequential();

      // Add one dense layer with:
      // - 1 neuron (units: 1)
      // - input expects 1 number (inputShape: [1])
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

      // 2. Compile the model
      // loss -> how wrong the model is
      // optimizer -> how the model updates weights while learning
      model.compile({
        loss: "meanSquaredError",
        optimizer: "sgd",
      });

      // 3. Training data (linear pattern y = 2x + 1)
      // xs = inputs, ys = outputs
      const xs = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const ys = tf.tensor([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]);

      console.log("Training...");

      // 4. Train the model
      await model.fit(xs, ys, {
        epochs: 30,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch} — Loss: ${logs.loss}`);
          },
        },
      });

      // Save the trained model so we can use it later
      modelRef.current = model;
    }

    run();
  }, []); // empty array = run once only

  // =======================================
  // HANDLE USER NUMBER INPUT
  // =======================================
  function handleInputUpload(e) {
    const val = Number(e.target.value);

    // Prevent invalid values (NaN)
    if (isNaN(val)) return;

    setUserNumber(val);
  }

  // =======================================
  // PREDICT OUTPUT USING TRAINED MODEL
  // =======================================
  function classifyNumber() {
    // If model not ready or no user input, do nothing
    if (!modelRef.current || userNumber === null) return;

    // Convert number into a tensor (model expects tensors)
    const inputTensor = tf.tensor([userNumber]);

    // Predict returns a Tensor, so extract value with dataSync()
    const output = modelRef.current.predict(inputTensor).dataSync()[0];

    // Round the result for display
    setPrediction(output.toFixed(2));
  }

  // =======================================
  // RESET EVERYTHING
  // =======================================
  function resetAll() {
    setUserNumber(null);
    setPrediction(null);
  }

  // =======================================
  // UI SECTION
  // =======================================
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6">Day 8 — First Neural Network</h1>
      <p className="text-lg font-semibold mb-4">Using user input</p>

      <input
        type="number"
        placeholder="Enter a Number"
        onChange={handleInputUpload}
        className="mb-4 border-2 p-2 rounded-lg"
      />

      {/* Show the user's input */}
      {userNumber !== null && <p className="mb-4">Your input: {userNumber}</p>}

      {/* Trigger prediction */}
      <button
        onClick={classifyNumber}
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 mb-4"
      >
        Predict Output
      </button>

      {/* Display model prediction */}
      {prediction && (
        <p className="text-xl font-bold mb-4">Model Output: {prediction}</p>
      )}

      {/* Reset button */}
      <button
        onClick={resetAll}
        className="bg-red-600 text-white px-6 py-2 rounded hover:bg-red-700"
      >
        Reset
      </button>
    </div>
  );
}
