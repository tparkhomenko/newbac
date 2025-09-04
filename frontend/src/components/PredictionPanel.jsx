import React from 'react'

const PredictionPanel = ({ prediction }) => {
  const { is_skin, lesion_type, malignancy, route_taken, metadata } = prediction

  const abbrevToFull = {
    // Melanocytic lesions
    NV: 'nevus',
    MEL: 'melanoma',
    
    // Non-melanocytic carcinomas  
    BCC: 'basal_cell_carcinoma',
    SCC: 'squamous_cell_carcinoma',
    
    // Keratosis lesions
    BKL: 'seborrheic_keratosis',
    AKIEC: 'actinic_keratosis',
    AK: 'actinic_keratosis',
    
    // Other lesions
    DF: 'dermatofibroma',
    VASC: 'vascular',
    
    // Special cases
    UNKNOWN: 'unknown',
    UNK: 'unknown',
    NOT_SKIN: 'not_skin',
    OTHER: 'other'
  }

  const gtAbbrev = metadata?.unified_diagnosis || null
  const groundTruth = gtAbbrev ? (abbrevToFull[gtAbbrev] || gtAbbrev) : 'N/A'

  // Check if ground truth indicates NOT_SKIN
  const isGroundTruthNotSkin = groundTruth === 'not_skin' || gtAbbrev === 'NOT_SKIN'

  const ConfidenceBar = ({ value, label, color = 'accent2' }) => (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="font-medium text-gray-700 dark:text-gray-300">{label}</span>
        <span className="text-gray-600 dark:text-gray-300">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-neutral-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            color === 'accent1' ? 'bg-accent1-600' : 'bg-accent2-600'
          }`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )

  const LesionTypeCard = ({ lesionType }) => (
    <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Lesion Type</h3>

      <div className="space-y-4">
        <div className="flex items-center justify-between p-3 bg-neutral-100 rounded-lg border border-neutral-300">
          <div className="text-sm">
            <div className="font-medium text-accent1-700">
              Prediction: {lesionType.labels[lesionType.label_index]} (Confidence: {(lesionType.confidence * 100).toFixed(1)}%)
            </div>
            <div className="text-gray-700 dark:text-gray-300">Ground Truth: {groundTruth}</div>
          </div>
        </div>

        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">All Probabilities:</h4>
          {lesionType.labels.map((label, index) => (
            <div key={label} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="font-medium text-gray-700 dark:text-gray-300">{label}</span>
                <span className="text-gray-600 dark:text-gray-300">
                  {(lesionType.probabilities[index] * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-neutral-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    index === lesionType.label_index ? 'bg-accent2-600' : 'bg-neutral-300'
                  }`}
                  style={{ width: `${lesionType.probabilities[index] * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )

  const MalignancyCard = ({ malignancy }) => (
    <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-primary mb-4">Malignancy</h3>

      <div className="space-y-4">
        <div className={`flex items-center justify-between p-3 rounded-lg border bg-neutral-100 border-neutral-300`}>
          <div className="text-sm">
            <div className={`font-medium text-accent1-700`}>
              Prediction: {malignancy.label.toUpperCase()} (Confidence: {(malignancy.confidence * 100).toFixed(1)}%)
            </div>
            <div className={`text-gray-700 dark:text-gray-300`}>
              Ground Truth: {groundTruth}
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Probabilities:</h4>
          <ConfidenceBar value={malignancy.probabilities.benign} label="Benign" color="accent2" />
          <ConfidenceBar value={malignancy.probabilities.malignant} label="Malignant" color="accent2" />
        </div>
      </div>
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Skin Detection */}
      <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-primary mb-4">Skin Detection</h3>

        <div className="space-y-4">
          <div className={`flex items-center justify-between p-3 rounded-lg border bg-neutral-100 border-neutral-300`}>
            <div className="text-sm">
              <div className="font-medium text-accent1-700">
                Prediction: {is_skin.label === 'skin' ? 'Skin' : 'Not Skin'} (Confidence: {(is_skin.confidence * 100).toFixed(1)}%)
              </div>
              <div className="text-gray-700 dark:text-gray-300">
                Ground Truth: {isGroundTruthNotSkin ? 'Not Skin' : (groundTruth === 'N/A' ? 'N/A' : 'Skin')}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">Probabilities:</h4>
            <ConfidenceBar value={is_skin.probabilities.skin} label="Skin" color="accent2" />
            <ConfidenceBar value={is_skin.probabilities.not_skin} label="Not Skin" color="accent2" />
          </div>
        </div>
      </div>

      {/* Lesion Type (only if skin detected AND ground truth is not NOT_SKIN) */}
      {is_skin.label === 'skin' && lesion_type && !isGroundTruthNotSkin && (
        <LesionTypeCard lesionType={lesion_type} />
      )}

      {/* Malignancy (only if skin detected AND ground truth is not NOT_SKIN) */}
      {is_skin.label === 'skin' && malignancy && !isGroundTruthNotSkin && (
        <MalignancyCard malignancy={malignancy} />
      )}

      {/* Show message if ground truth is NOT_SKIN but model predicts skin */}
      {isGroundTruthNotSkin && is_skin.label === 'skin' && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">Ground Truth Mismatch</h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>Ground truth indicates this is NOT a skin lesion, but the model predicted it as skin. Lesion type and malignancy predictions are hidden.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline Route */}
      <div className="bg-neutral-100 dark:bg-gray-800 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Pipeline Route:</h4>
        <div className="flex flex-wrap gap-2">
          {route_taken.map((step, index) => (
            <span
              key={index}
              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-neutral-100 text-primary border border-neutral-300"
            >
              {step}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default PredictionPanel
