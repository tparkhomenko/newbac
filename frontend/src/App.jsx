import React, { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import PredictionPanel from './components/PredictionPanel'
import MetadataPanel from './components/MetadataPanel'
import ModelPanel from './components/ModelPanel'
// Placeholder: Quick Test feature removed from UI per requirement
// import QuickTestPanel from './components/QuickTestPanel'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [metadata, setMetadata] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [currentModel, setCurrentModel] = useState('mlp1.pt')

  const handlePrediction = (predictionData) => {
    setPrediction(predictionData)
    setMetadata(predictionData.metadata)
    setError(null)
  }

  const handleError = (errorMessage) => {
    setError(errorMessage)
    setPrediction(null)
    setMetadata(null)
  }

  const handleModelSwitch = (newModel) => {
    setCurrentModel(newModel)
    // Clear previous results when switching models
    setPrediction(null)
    setMetadata(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 dark:text-gray-100">
      {/* Header */}
      <header className="bg-neutral-100 dark:bg-gray-800 shadow-sm border-b border-neutral-300 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <span className="text-white text-lg font-bold">🏥</span>
                </div>
              </div>
              <div className="ml-3">
                <h1 className="text-xl font-semibold text-primary">
                  Skin Lesion Classifier
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  AI-powered skin lesion classification with model switching
                </p>
              </div>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Model: <span className="font-medium text-accent1-600">{currentModel}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Sidebar - Model Management */}
          <div className="lg:col-span-1 space-y-6">
            <ModelPanel onModelSwitch={handleModelSwitch} />
            {/* Placeholder: Quick Test panel hidden per requirement */}
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-3 space-y-6">
            {/* Image Upload Section */}
            <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-primary mb-4">
                📸 Upload Image
              </h2>
              <ImageUploader
                onPrediction={handlePrediction}
                onError={handleError}
                onLoading={setLoading}
              />
            </div>

            {/* Results Section */}
            {loading && (
              <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent2-600"></div>
                  <span className="ml-3 text-gray-600 dark:text-gray-300">Processing image...</span>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">Error</h3>
                    <div className="mt-2 text-sm text-red-700">{error}</div>
                  </div>
                </div>
              </div>
            )}

            {prediction && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Predictions */}
                <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-primary mb-4">
                    🔍 Predictions
                  </h2>
                  <PredictionPanel prediction={prediction} />
                </div>

                {/* Metadata */}
                <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-primary mb-4">
                    📊 Image Metadata
                  </h2>
                  <MetadataPanel metadata={metadata} />
                </div>
              </div>
            )}

            {/* Welcome Message */}
            {!prediction && !loading && !error && (
              <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-12 text-center">
                <div className="mx-auto h-24 w-24 text-gray-400 mb-4">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-primary mb-2">
                  Ready to analyze skin lesions
                </h3>
                <p className="text-gray-600 dark:text-gray-300 max-w-sm mx-auto">
                  Upload an image to get AI-powered predictions for skin lesion classification.
                  {/* Placeholder: removed reference to Quick Test from welcome text */}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
