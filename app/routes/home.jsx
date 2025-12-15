import { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export function meta() {
  return [
    { title: "TensorFlow in ReactRouter" },
    { name: "description", content: "Welcome to Learning TensorFow.js!" },
  ];
}

export default function Home() {
  useEffect(() => {
    const tensor = tf.tensor([1, 2, 3, 4]);
    console.log("My first tensor:", tensor);
    console.log("Shape:", tensor.shape);

    tensor.print(); // Shows actual values
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-2xl font-bold">Tensor Created ✅</h1>
      <p className="text-lg font-bold mb-6">
        Open your console to see training logs and prediction output.
      </p>
    </div>
  );
}
