import { useEffect, useRef, useState } from "react"; // React hooks
import * as mobilenet from "@tensorflow-models/mobilenet"; // Pre-trained ML model
import * as tf from "@tensorflow/tfjs"; // TensorFlow.js library

export default function Home() {
  // ------------------------
  // State variables
  // ------------------------
  let [imageSrc, setImageSrc] = useState(null); // Stores uploaded image as Base64
  let [predictions, setPredictions] = useState([]); // Stores predictions from model
  let [model, setModel] = useState(null); // Stores loaded MobileNet model
  let imgRef = useRef(); // Reference to the <img> element
  const [loadingModel, setLoadingModel] = useState(true); // Loading indicator

  // ------------------------
  // Load the MobileNet model on component mount
  // ------------------------
  useEffect(() => {
    async function loadModel() {
      try {
        console.log("Initializing TFJS");
        await tf.ready(); // Wait until TensorFlow.js is ready
        await tf.setBackend("webgl"); // Use GPU for faster computations
        console.log("Loading model...");
        let loaded = await mobilenet.load(); // Load pre-trained MobileNet model
        console.log("✅ Model loaded");
        setModel(loaded); // Save model to state
        setLoadingModel(false); // mark model as loaded
      } catch (error) {
        // Handle network errors or loading issues
        console.error("❌ Model failed to load:", error);
      }
    }
    loadModel();
  }, []); // Empty dependency array = run once when component mounts

  // ------------------------
  // Handle image file upload
  // ------------------------
  let handleImageUpload = (e) => {
    const file = e.target.files[0]; // Get first file
    if (!file) return; // Stop if no file selected

    const reader = new FileReader(); // Create FileReader to read file

    reader.onload = () => {
      // Runs when reading is complete
      setImageSrc(reader.result); // Save Base64 string to state
    };

    reader.readAsDataURL(file); // Read file as Base64 URL
  };

  // ------------------------
  // Reset uploaded image and predictions
  // ------------------------
  const reset = () => {
    setImageSrc(null); // Clear image
    setPredictions([]); // Clear predictions
  };

  // ------------------------
  // Optional: another useEffect to track loading state
  // ------------------------
  useEffect(() => {
    mobilenet.load().then((loaded) => {
      setModel(loaded);
      setLoadingModel(false); // Model finished loading
    });
  }, []);

  // ------------------------
  // Run the model on the uploaded image
  // ------------------------
  let classifyImage = async () => {
    if (!model || !imgRef.current) return; // Stop if model or image not ready
    let results = await model.classify(imgRef.current); // Predict image class
    console.log("🔍 Running prediction...");
    setPredictions(results); // Save predictions to state
  };

  // ------------------------
  // Render JSX
  // ------------------------
  return (
    <div className="min-h-screen flex flex-col items-center justify-center  p-6">
      <h1 className="text-3xl font-bold mb-6">Ai Image Classifier</h1>
      <p className="text-lg font-bold mb-6">
        Using user input to classify their images as either cats and dogs
      </p>

      {/* Show loading message while model is loading */}
      {loadingModel && <p>Loading model… please wait</p>}

      {/* File input for image upload */}
      <input
        type="file"
        accept="image/*"
        required
        className="mb-4 border-2 p-1 rounded-lg"
        onChange={handleImageUpload} // Calls handleImageUpload on file selection
      />

      {/* Display uploaded image */}
      {imageSrc && (
        <img
          ref={imgRef} // Reference used for model classification
          src={imageSrc}
          alt="Upload Preview"
          className="w-64 h-64 object-cover rounded shadow mb-4"
        />
      )}

      {/* Button to classify image */}
      <button
        onClick={classifyImage} // Runs MobileNet classification
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 mb-4 disabled:opacity-50"
        disabled={!imageSrc} // Disable button if no image
      >
        Classify Image
      </button>

      {/* Button to reset image and predictions */}
      <button
        onClick={reset}
        disabled={!imageSrc} // Disable if no image
        className="bg-red-600 text-white px-6 py-2 rounded hover:bg-red-700 mb-4 disabled:opacity-50"
      >
        Reset The Image
      </button>

      {/* Display predictions */}
      {predictions.length > 0 && (
        <div className="bg-white p-4 rounded shadow w-full max-w-md">
          <h2 className="text-xl font-semibold mb-2">Predictions:</h2>
          {predictions.map((p, index) => (
            <div
              key={index}
              className="flex justify-between bg-white p-4 rounded shadow mb-2"
            >
              <span>{p.className}</span> {/* Predicted label */}
              <span>{(p.probability * 100).toFixed(2)}%</span>{" "}
              {/* Confidence */}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
