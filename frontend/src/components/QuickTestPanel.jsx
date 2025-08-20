import React, { useState } from 'react'

const QuickTestPanel = () => {
  const [testResults, setTestResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runQuickTest = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await fetch('http://127.0.0.1:8000/quicktest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setTestResults(data)
    } catch (err) {
      setError(`Failed to run quick test: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`
  }

  const formatTime = (seconds) => {
    if (seconds < 1) {
      return `${(seconds * 1000).toFixed(0)}ms`
    }
    return `${seconds.toFixed(2)}s`
  }

  const renderConfusionMatrix = (matrix, classNames) => {
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-gray-600 dark:text-gray-300">Predicted →</th>
              {classNames.map((name, idx) => (
                <th key={idx} className="px-2 py-1 text-center text-gray-600 dark:text-gray-300">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIdx) => (
              <tr key={rowIdx}>
                <td className="px-2 py-1 text-left text-gray-600 dark:text-gray-300 font-medium">
                  {classNames[rowIdx]} (Actual)
                </td>
                {row.map((cell, colIdx) => (
                  <td
                    key={colIdx}
                    className={`px-2 py-1 text-center ${
                      rowIdx === colIdx
                        ? 'bg-accent2-600/20 text-accent2-700 font-medium'
                        : 'bg-neutral-100 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-primary">
          🧪 Quick Test
        </h3>
        <button
          onClick={runQuickTest}
          disabled={loading}
          className="px-4 py-2 bg-accent1-600 text-white text-sm font-medium rounded-md hover:bg-accent1-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-accent1-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Running Test...
            </div>
          ) : (
            'Run Quick Test'
          )}
        </button>
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
        Test the current model on all images in the testing directory (up to 100 images).
        This will evaluate accuracy, F1 score, and generate a confusion matrix.
      </p>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
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

      {testResults && (
        <div className="space-y-6">
          {/* Summary Stats */}
          <div className="bg-neutral-100 dark:bg-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-200 mb-3">Test Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-accent2-700">
                  {testResults.total_images}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Images Tested</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent2-700">
                  {formatPercentage(testResults.accuracy)}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Overall Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent2-700">
                  {formatPercentage(testResults.f1_score)}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">F1 Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent2-700">
                  {formatTime(testResults.test_time)}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-300">Test Time</div>
              </div>
            </div>
            <div className="mt-3 text-xs text-gray-500 dark:text-gray-300 text-center">
              Model used: {testResults.model_used}
            </div>
          </div>

          {/* Per-Class Accuracy */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-200 mb-3">Per-Class Accuracy</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {Object.entries(testResults.per_class_accuracy).map(([className, accuracy]) => (
                <div key={className} className="bg-neutral-100 dark:bg-gray-700 rounded p-3 text-center">
                  <div className="text-lg font-bold text-accent2-700">
                    {formatPercentage(accuracy)}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-300">{className}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Confusion Matrix */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-200 mb-3">Confusion Matrix</h4>
            <div className="bg-neutral-100 dark:bg-gray-700 rounded-lg p-4">
              {renderConfusionMatrix(
                testResults.confusion_matrix,
                ['melanoma', 'nevus', 'seborrheic_keratosis', 'basal_cell_carcinoma', 'actinic_keratosis']
              )}
            </div>
            <p className="mt-2 text-xs text-gray-500 dark:text-gray-300">
              Rows: Actual class, Columns: Predicted class. Green cells show correct predictions.
            </p>
          </div>
        </div>
      )}

      {!testResults && !loading && (
        <div className="text-center py-8 text-gray-500 dark:text-gray-300">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="mt-2 text-sm">Click "Run Quick Test" to evaluate the current model</p>
        </div>
      )}
    </div>
  )
}

export default QuickTestPanel
