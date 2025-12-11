import { useEffect, useRef, useState } from "react";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";
export default function Home() {
  const [model, setModel] = useState(null);
  let [predictions, setPredictions] = useState([]);
  const imgRef = useRef();

  useEffect(() => {
    async function loadModel() {
      try {
        console.log("Initializing TFJS...");
        await tf.ready();
        await tf.setBackend("webgl"); // or "cpu"
        console.log("Loading model...");
        const loaded = await mobilenet.load();
        console.log("✅ Model loaded");
        setModel(loaded);
      } catch (error) {
        // most likely to arise because of slow networks
        console.error("❌ Model failed to load:", error);
      }
    }

    loadModel();
  }, []);

  const classifyImage = async () => {
    if (!model) return console.log("❌ Model not loaded");
    if (!imgRef.current) return console.log("❌ Image not ready");

    console.log("🔍 Running prediction...");
    const results = await model.classify(imgRef.current);
    console.log("✅ Predictions:", results);
    setPredictions(results);
  };
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-4">
        TensorFlow.js Image Classifier
      </h1>
      <p>We will look on how to classify images using a pretrained model</p>
      <span>We shall use an import @tensorflow-models/mobilenet</span>
      <img
        ref={imgRef}
        src="https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg"
        //
        alt="Sample"
        crossOrigin="anonymous"
        className="w-64 h-64 object-cover mb-4"
        onLoad={() => console.log("✅ Image loaded")}
      />

      <button
        onClick={classifyImage}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Classify Image
      </button>

      {predictions.length > 0 && (
        <div className="bg-white p-4 rounded shadow w-full max-w-md">
          <h2 className="text-xl font-semibold mb-2">Predictions:</h2>
          {predictions.map((p, index) => (
            <div key={index} className="flex justify-between">
              <span>{p.className}</span>
              <span>{(p.probability * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
