import * as tf from "@tensorflow/tfjs";
import { useEffect, useRef, useState } from "react";

// show if is > 10 true/false
export default function GreaterThanTenClassifier() {
  let [userNumber, setUserNumber] = useState(null);
  let [training, setTrainingModel] = useState(true);
  const [result, setResult] = useState(null);
  const [prediction, setPrediction] = useState(null);

  let modelRef = useRef(null);

  useEffect(() => {
    async function train() {
      let model = tf.sequential();
      model.add(
        tf.layers.dense({
          units: 1,
          inputShape: [1],
          activation: "sigmoid",
        })
      );
      model.compile({
        loss: "binaryCrossentropy",
        optimizer: "adam",
      });
      let xs = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]);
      let ys = tf.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
      await model.fit(xs, ys, {
        epochs: 50,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch} — Loss: ${logs.loss}`);
          },
        },
      });
      console.log("model trained");
      setTrainingModel(false);
      modelRef.current = model;
    }
    train();
  }, []);

  function predict() {
    if (!modelRef.current || userNumber === null) return;
    let output = modelRef.current
      .predict(tf.tensor([userNumber]))
      .dataSync()[0];
    console.log("output", output);
    setResult(
      output > 0.5
        ? `yes ${userNumber} is greater than 10`
        : `No ${userNumber} is not greater than 10`
    );
    // Round the result for display
    setPrediction((output * 100).toFixed(2));
  }
  function handleChange(e) {
    let val = Number(e.target.value);
    if (isNaN(val)) return;
    setUserNumber(val);
  }

  function reset() {
    setUserNumber(null);
    setPrediction(null);
    setResult(null);
  }
  return (
    <div className="min-h-screen flex flex-col items-center justify-center  p-6">
      <h1 className="text-xl  font-bold">Day 9 — Binary Classification</h1>
      <p className="text-xl  font-bold">
        Decide if a Number is greater than 10
      </p>
      {training && (
        <p className="text-blue-600 italic mb-2">Model Training… please wait</p>
      )}
      <input
        type="number"
        onChange={handleChange}
        className="border p-2 my-3"
      />

      {/* Show the user's input */}
      {userNumber !== null && <p className="mb-4">Your input: {userNumber}</p>}
      <button
        onClick={predict}
        disabled={training}
        className=" bg-blue-600 text-white px-4 py-2 rounded "
      >
        Classify
      </button>

      <button onClick={reset} className="border p-2 my-3">
        Reset
      </button>
      {result && <p className="mt-4 font-bold">{result}</p>}
      {/* Display model prediction */}
      {prediction && (
        <p className="text-xl font-bold mb-4">Model Output: {prediction}%</p>
      )}
    </div>
  );
}
