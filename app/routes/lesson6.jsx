import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function EvenOddClassifier() {
  const [userNumber, setUserNumber] = useState(null);
  let [training, setTrainingModel] = useState(true);
  const [result, setResult] = useState(null);
  const modelRef = useRef(null);
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

      let xs = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      let ys = tf.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
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
      .dataSync()[0]; //Converts the tensor into a TypedArray //  Takes the first value from the array
    console.log("output", output);
    setResult(
      output > 0.5 ? `yes ${userNumber} is even` : `No ${userNumber} is odd`
    );
  }
  return (
    <div className="min-h-screen flex flex-col items-center justify-center  p-6">
      <h1 className="text-xl  font-bold">Day 9 — Binary Classification</h1>
      <p className="text-xl  font-bold">Determining is Even or Odd</p>
      {training && (
        <p className="text-blue-600 italic mb-2">Model Training… please wait</p>
      )}
      <input
        type="number"
        onChange={(e) => setUserNumber(Number(e.target.value))}
        className="border p-2 my-3"
      />

      {/* Show the user's input */}
      {userNumber !== null && <p className="mb-4">Your input: {userNumber}</p>}
      <button
        onClick={predict}
        className="bg-blue-600 text-white px-4 py-2 rounded"
      >
        Classify
      </button>

      {result && <p className="mt-4 font-bold">{result}</p>}
    </div>
  );
}
