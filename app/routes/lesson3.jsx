import { useEffect, useRef, useState } from "react";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";

export default function Home() {
  let [imageSrc, setImageSrc] = useState(null);
  let [predictions, setPredictions] = useState([]);
  let [model, setModel] = useState(null);
  let imgRef = useRef();

  useEffect(() => {
    async function loadModel() {
      try {
        console.log("Initializing TFJS");
        await tf.ready(); //wait till the model is loaded
        await tf.setBackend("webgl"); //chose model
        console.log("Loading model...");
        let loaded = await mobilenet.load();
        console.log("✅ Model loaded");
        setModel(loaded);
      } catch (error) {
        // most likely to arise because of slow networks
        console.error("❌ Model failed to load:", error);
      }
    }
    loadModel();
  }, []);

  const handleImageUpload = (e) => {
    // Get the first selected file from the input element
    const file = e.target.files[0];

    // If no file was selected, stop the function
    if (!file) return;

    // Create a FileReader to read the file's contents
    const reader = new FileReader();

    // This runs when FileReader finishes reading the file
    reader.onload = () => {
      // Save the image (as a Base64 string) into React state
      // This allows us to display the image or use it in TensorFlow.js
      setImageSrc(reader.result);
    };

    // Start reading the file as a Data URL (Base64 encoded image)
    reader.readAsDataURL(file);
  };

  let classifyImage = async () => {
    if (!model || !imgRef.current) return;
    let results = await model.classify(imgRef.current);
    console.log("🔍 Running prediction...");
    setPredictions(results);
  };
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6">Ai Image Classifier</h1>
      <p className="text-lg font-bold mb-6">
        using user input to classify their images as either cats and Dogs
      </p>
      <input
        type="file"
        accept="image/*"
        required
        className="mb-4 border-2 p-1 rounded-lg"
        onChange={handleImageUpload}
      />
      {imageSrc && (
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Upload Preview"
          className="w-64 h-64 object-cover rounded shadow mb-4"
        />
      )}

      <button
        onClick={classifyImage}
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 mb-4 disabled:opacity-50"
        disabled={!imageSrc}
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
