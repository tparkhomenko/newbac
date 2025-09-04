import React, { useState, useEffect } from 'react'

const ModelPanel = ({ onModelSwitch }) => {
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchModelInfo()
  }, [])

  const fetchModelInfo = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://127.0.0.1:8000/model')
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setModelInfo(data)
      setError(null)
    } catch (err) {
      setError(`Failed to fetch model info: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const switchTo = async (choice) => {
    try {
      setLoading(true)
      const response = await fetch('http://127.0.0.1:8000/model/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_choice: choice })
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setModelInfo(prev => ({
        ...(prev || {}),
        current_model: data.current_model,
        actual_model: data.actual_model,
        summary: data.summary || prev?.summary
      }))
      onModelSwitch && onModelSwitch(data.current_model)
      setError(null)
    } catch (err) {
      setError(`Failed to switch model: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const formatPercentage = (value) => {
    if (value === null || value === undefined) return '—'
    return `${(value * 100).toFixed(2)}%`
  }

  if (loading && !modelInfo) {
    return (
      <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-accent2-600"></div>
          <span className="ml-3 text-gray-600 dark:text-gray-300">Loading model info...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
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
            <button onClick={fetchModelInfo} className="mt-2 text-sm text-red-600 hover:text-red-500 underline">Retry</button>
          </div>
        </div>
      </div>
    )
  }

  if (!modelInfo) return null

  const isActive = (choice) => modelInfo.current_model === choice

  return (
    <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Model</h3>

      {/* Current Model */}
      <div className="mb-4 text-sm">
        <span className="text-gray-600 dark:text-gray-300">Active:</span>
        <span className="ml-2 font-medium text-accent1-600">{modelInfo.actual_model || modelInfo.current_model}</span>
      </div>

      {/* Toggle Buttons for best models */}
      <div className="flex gap-2 mb-6">
        {['parallel_best', 'multi_best'].map((choice) => (
          <button
            key={choice}
            onClick={() => switchTo(choice)}
            disabled={loading}
            className={`px-4 py-2 rounded-md text-sm font-medium border ${
              isActive(choice)
                ? 'bg-accent1-600 text-white border-accent1-600'
                : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 border-neutral-300 hover:bg-neutral-100'
            }`}
          >
            {choice === 'parallel_best' ? 'Parallel Best' : 'Multi Best'}
          </button>
        ))}
      </div>

      {/* Summary stats */}
      {modelInfo.summary && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Model Stats</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-neutral-100 dark:bg-gray-700 rounded p-2">
              <div className="text-gray-600 dark:text-gray-300">Skin Acc</div>
              <div className="font-bold text-accent2-700">{formatPercentage(modelInfo.summary?.skin?.acc)}</div>
            </div>
            <div className="bg-neutral-100 dark:bg-gray-700 rounded p-2">
              <div className="text-gray-600 dark:text-gray-300">Skin F1</div>
              <div className="font-bold text-accent2-700">{formatPercentage(modelInfo.summary?.skin?.f1)}</div>
            </div>
            <div className="bg-neutral-100 dark:bg-gray-700 rounded p-2">
              <div className="text-gray-600 dark:text-gray-300">Lesion Acc</div>
              <div className="font-bold text-accent2-700">{formatPercentage(modelInfo.summary?.lesion?.acc)}</div>
            </div>
            <div className="bg-neutral-100 dark:bg-gray-700 rounded p-2">
              <div className="text-gray-600 dark:text-gray-300">Lesion F1</div>
              <div className="font-bold text-accent2-700">{formatPercentage(modelInfo.summary?.lesion?.f1)}</div>
            </div>
            <div className="bg-neutral-100 dark:bg-gray-700 rounded p-2">
              <div className="text-gray-600 dark:text-gray-300">B/M Acc</div>
              <div className="font-bold text-accent2-700">{formatPercentage(modelInfo.summary?.bm?.acc)}</div>
            </div>
            <div className="bg-neutral-100 dark:bg-gray-700 rounded p-2">
              <div className="text-gray-600 dark:text-gray-300">B/M F1</div>
              <div className="font-bold text-accent2-700">{formatPercentage(modelInfo.summary?.bm?.f1)}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelPanel
