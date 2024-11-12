import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Brain, Activity, Droplets, Calendar, TrendingUp } from 'lucide-react';
import { format, addDays } from 'date-fns';
import * as tf from '@tensorflow/tfjs';

interface BloodDemand {
  date: string;
  actual: number;
  predicted: number;
}

function App() {
  const [historicalData, setHistoricalData] = useState<BloodDemand[]>([]);
  const [predictions, setPredictions] = useState<BloodDemand[]>([]);
  const [loading, setLoading] = useState(true);

  // Simulate historical data
  useEffect(() => {
    const generateData = () => {
      const data: BloodDemand[] = [];
      const baseValue = 100;
      const seasonalFactor = 20;
      const trendFactor = 0.5;

      for (let i = 30; i >= 0; i--) {
        const date = addDays(new Date(), -i);
        const seasonal = Math.sin((i / 7) * Math.PI) * seasonalFactor;
        const trend = i * trendFactor;
        const random = Math.random() * 10 - 5;
        const value = Math.max(0, Math.round(baseValue + seasonal + trend + random));

        data.push({
          date: format(date, 'MMM dd'),
          actual: value,
          predicted: 0
        });
      }
      return data;
    };

    const data = generateData();
    setHistoricalData(data);

    // Simple prediction model using TensorFlow.js
    const trainModel = async () => {
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 1, inputShape: [7] }));
      model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

      // Prepare training data
      const values = data.map(d => d.actual);
      const X = [];
      const y = [];
      for (let i = 7; i < values.length; i++) {
        X.push(values.slice(i - 7, i));
        y.push(values[i]);
      }

      const xs = tf.tensor2d(X);
      const ys = tf.tensor2d(y, [y.length, 1]);

      await model.fit(xs, ys, { epochs: 100 });

      // Generate predictions
      const futurePredictions: BloodDemand[] = [];
      let lastWindow = values.slice(-7);

      for (let i = 1; i <= 7; i++) {
        const prediction = model.predict(tf.tensor2d([lastWindow])) as tf.Tensor;
        const predictedValue = Math.round(prediction.dataSync()[0]);
        const date = addDays(new Date(), i);
        
        futurePredictions.push({
          date: format(date, 'MMM dd'),
          actual: 0,
          predicted: predictedValue
        });

        lastWindow = [...lastWindow.slice(1), predictedValue];
      }

      setPredictions(futurePredictions);
      setLoading(false);
    };

    trainModel();
  }, []);

  const allData = [...historicalData, ...predictions];

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex items-center mb-6">
            <Brain className="w-8 h-8 text-indigo-600 mr-3" />
            <h1 className="text-2xl font-bold text-gray-900">Blood Products Demand Forecasting</h1>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-indigo-50 p-6 rounded-lg">
                  <div className="flex items-center mb-2">
                    <Activity className="w-5 h-5 text-indigo-600 mr-2" />
                    <h3 className="text-lg font-semibold text-gray-900">Current Demand</h3>
                  </div>
                  <p className="text-3xl font-bold text-indigo-600">
                    {historicalData[historicalData.length - 1]?.actual} units
                  </p>
                </div>

                <div className="bg-green-50 p-6 rounded-lg">
                  <div className="flex items-center mb-2">
                    <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
                    <h3 className="text-lg font-semibold text-gray-900">Predicted Peak</h3>
                  </div>
                  <p className="text-3xl font-bold text-green-600">
                    {Math.max(...predictions.map(d => d.predicted))} units
                  </p>
                </div>

                <div className="bg-blue-50 p-6 rounded-lg">
                  <div className="flex items-center mb-2">
                    <Calendar className="w-5 h-5 text-blue-600 mr-2" />
                    <h3 className="text-lg font-semibold text-gray-900">Forecast Period</h3>
                  </div>
                  <p className="text-3xl font-bold text-blue-600">7 days</p>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 mb-8">
                <h2 className="text-xl font-semibold mb-4">Demand Forecast Chart</h2>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={allData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="actual" 
                        stroke="#4F46E5" 
                        name="Actual Demand"
                        strokeWidth={2}
                        dot={{ r: 2 }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="predicted" 
                        stroke="#059669" 
                        name="Predicted Demand"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={{ r: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-white rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <Droplets className="w-6 h-6 text-red-600 mr-2" />
                  <h2 className="text-xl font-semibold">Recommendations</h2>
                </div>
                <ul className="space-y-3 text-gray-700">
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-indigo-600 rounded-full mr-2"></span>
                    Increase collection by {Math.round(predictions[0].predicted - historicalData[historicalData.length - 1].actual)} units for tomorrow
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-indigo-600 rounded-full mr-2"></span>
                    Schedule additional donation drives for next week
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-indigo-600 rounded-full mr-2"></span>
                    Monitor O-negative blood type stocks closely
                  </li>
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;