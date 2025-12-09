import { useEffect } from "react";
import { Welcome } from "../welcome/welcome";
import * as tf from "@tensorflow/tfjs";

export function meta() {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
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
    <div className="min-h-screen flex items-center justify-center">
      <h1 className="text-2xl font-bold">Tensor Created ✅</h1>
    </div>
  );
}
