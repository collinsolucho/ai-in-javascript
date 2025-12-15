# Learning Project: React Router + TensorFlow.js

This project builds on the React Router template and adds a simple TensorFlow.js demo to explore tensors and pre-trained models like MobileNet.

## What the app does

- Demonstrates basic tensor creation and inspection in the home route:

```startLine:endLine:app/routes/home.jsx
import { useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

export default function Home() {
  useEffect(() => {
    const tensor = tf.tensor([1, 2, 3, 4]);
    console.log("My first tensor:", tensor);
    console.log("Shape:", tensor.shape);
    tensor.print();
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <h1 className="text-2xl font-bold">Tensor Created ✅</h1>
    </div>
  );
}
```

- Notes explored so far:
  - `tf.tensor` converts input to tensor data; use inside `useEffect` for post-render side effects.
  - Shapes (`tensor.shape`) show dimensions like `[rows, columns]`; `tensor.size` is the total element count.
  - 2D tensors are matrices; 3D tensors add depth (e.g., `[depth, rows, columns]`).
  - `await tf.ready()` ensures TensorFlow.js is loaded; `tf.setBackend("webgl" | "cpu" | "wasm")` selects the compute engine (webgl is fastest for ML in-browser).
  - MobileNet can be added for simple image predictions (`@tensorflow-models/mobilenet`).

## Prerequisites

- Node.js 18+
- npm

## Installation & setup

```bash
# 1) create the project (already done here, for reference)
mkdir <folderName>
cd <folderName>
npx create-react-router@latest --template remix-run/react-router-templates/javascript

# 2) install dependencies
npm install

# 3) add TensorFlow.js (client) and MobileNet
npm install @tensorflow/tfjs @tensorflow-models/mobilenet
```

## Running the app

```bash
npm run dev
# open http://localhost:5173
```

## Building for production

```bash
npm run build
```

## Deployment

- Docker example:

  ```bash
  docker build -t my-app .
  docker run -p 3000:3000 my-app
  ```

- Deploy the output of `npm run build` (`build/client` and `build/server`) to your platform of choice (ECS, Cloud Run, Azure Container Apps, Fly.io, Railway, etc.).

## Styling

Tailwind CSS is preconfigured, but you can use any styling approach you prefer.
