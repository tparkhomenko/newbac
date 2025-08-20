import React from 'react'

const MetadataPanel = ({ metadata }) => {
  if (!metadata) return null

  const getExperimentStatus = (expValue) => {
    if (!expValue || expValue === '') return { status: 'not_included', label: 'Not Included' }
    if (expValue === 'train') return { status: 'train', label: 'Training Set' }
    if (expValue === 'val') return { status: 'val', label: 'Validation Set' }
    if (expValue === 'test') return { status: 'test', label: 'Test Set' }
    return { status: 'other', label: expValue }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'train':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'val':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'test':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'not_included':
        return 'bg-gray-100 text-gray-800 border-gray-200'
      default:
        return 'bg-blue-100 text-blue-800 border-blue-200'
    }
  }

  const getDiagnosisColor = (diagnosis) => {
    const diagnosisColors = {
      'NV': 'bg-blue-100 text-blue-800 border-blue-200',      // Nevus
      'MEL': 'bg-red-100 text-red-800 border-red-200',       // Melanoma
      'BCC': 'bg-orange-100 text-orange-800 border-orange-200', // Basal Cell Carcinoma
      'AK': 'bg-purple-100 text-purple-800 border-purple-200',  // Actinic Keratosis
      'SK': 'bg-green-100 text-green-800 border-green-200',   // Seborrheic Keratosis
    }
    return diagnosisColors[diagnosis] || 'bg-gray-100 text-gray-800 border-gray-200'
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        📊 Dataset Metadata
      </h3>
      
      <div className="space-y-4">
        {/* Image Name */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm font-medium text-gray-700">Image Name</div>
          <div className="text-lg font-mono text-gray-900">{metadata.image_name}</div>
        </div>

        {/* Source and Diagnosis */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">Source CSV</div>
            <div className="text-sm text-gray-900 truncate">{metadata.csv_source}</div>
          </div>
          
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-sm font-medium text-gray-700">Original Diagnosis</div>
            <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getDiagnosisColor(metadata.diagnosis_from_csv)}`}>
              {metadata.diagnosis_from_csv}
            </div>
          </div>
        </div>

        {/* Unified Diagnosis */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-sm font-medium text-gray-700">Unified Diagnosis</div>
          <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-medium border ${getDiagnosisColor(metadata.unified_diagnosis)}`}>
            {metadata.unified_diagnosis}
          </div>
        </div>

        {/* Experiment Splits */}
        <div className="space-y-3">
          <div className="text-sm font-medium text-gray-700">Experiment Splits</div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {[
              { key: 'exp1', label: 'Exp 1' },
              { key: 'exp2', label: 'Exp 2' },
              { key: 'exp3', label: 'Exp 3' },
              { key: 'exp4', label: 'Exp 4' },
              { key: 'exp5', label: 'Exp 5' }
            ].map(({ key, label }) => {
              const { status, label: statusLabel } = getExperimentStatus(metadata[key])
              return (
                <div key={key} className="text-center">
                  <div className="text-xs font-medium text-gray-600 mb-1">{label}</div>
                  <div className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium border ${getStatusColor(status)}`}>
                    {statusLabel}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="pt-4 border-t border-gray-200">
          <div className="text-xs text-gray-600 mb-2">Legend:</div>
          <div className="flex flex-wrap gap-2 text-xs">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-100 border border-green-200 rounded mr-1"></div>
              <span className="text-gray-600">Training</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-yellow-100 border border-yellow-200 rounded mr-1"></div>
              <span className="text-gray-600">Validation</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-100 border border-red-200 rounded mr-1"></div>
              <span className="text-gray-600">Test</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-gray-100 border border-gray-200 rounded mr-1"></div>
              <span className="text-gray-600">Not Included</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MetadataPanel
