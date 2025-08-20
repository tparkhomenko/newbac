# 🏥 Skin Lesion Classifier Frontend

A React-based web demo for the Skin Lesion Classification system that integrates with the FastAPI backend.

## ✨ Features

- **Image Upload**: Drag & drop or click to upload images
- **Real-time Predictions**: Get instant ML pipeline results
- **Confidence Visualization**: Beautiful progress bars for all probabilities
- **Metadata Display**: Show dataset information and experiment splits
- **Responsive Design**: Works on desktop and mobile devices
- **Professional UI**: Clean, medical-themed interface

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- Backend server running on `http://127.0.0.1:8000`

### Installation

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:3000`

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUploader.jsx      # Image upload with drag & drop
│   │   ├── PredictionPanel.jsx    # ML results display
│   │   └── MetadataPanel.jsx      # Dataset metadata
│   ├── App.jsx                    # Main app component
│   ├── main.jsx                   # React entry point
│   └── index.css                  # Tailwind CSS imports
├── package.json                   # Dependencies and scripts
├── vite.config.js                 # Vite configuration
├── tailwind.config.js             # Tailwind CSS config
└── postcss.config.js              # PostCSS configuration
```

## 🔧 Configuration

### Backend URL

The frontend connects to the backend at `http://127.0.0.1:8000`. To change this:

1. Edit `src/components/ImageUploader.jsx`
2. Update the fetch URL in the `handleFile` function

### Styling

The app uses Tailwind CSS with a custom color scheme:

- **Primary**: Blue tones for main actions
- **Success**: Green for positive results
- **Warning**: Yellow for validation sets
- **Danger**: Red for test sets and malignant results

## 📱 Usage

### 1. Upload Image
- **Drag & Drop**: Drag an image file onto the upload area
- **Click Upload**: Click "Upload an image" to browse files
- **Supported Formats**: JPG, PNG, GIF, BMP, TIFF
- **Max Size**: 10MB

### 2. View Results
The system provides a 3-stage analysis:

1. **Skin Detection**: Is this a skin image?
2. **Lesion Classification**: What type of skin lesion? (5 classes)
3. **Malignancy Assessment**: Benign or malignant?

### 3. Metadata Information
For images in the dataset, you'll see:
- **Image Name**: Filename without extension
- **Source**: Original CSV source
- **Diagnosis**: Ground truth labels
- **Experiment Splits**: Training/validation/test assignments

## 🎨 UI Components

### ImageUploader
- Drag & drop interface
- File validation
- Image preview
- Upload progress

### PredictionPanel
- **Skin Detection**: Green for skin, gray for non-skin
- **Lesion Type**: Blue highlighting for predicted class
- **Malignancy**: Red for malignant, green for benign
- **Confidence Bars**: Visual probability representation
- **Pipeline Route**: Shows which models were used

### MetadataPanel
- **Color-coded Labels**: Different colors for diagnosis types
- **Experiment Status**: Visual indicators for dataset splits
- **Responsive Grid**: Adapts to screen size

## 🔌 API Integration

### Endpoints Used

- `GET /healthz` - Check backend status
- `POST /predict` - Submit image for analysis

### Request Format
```javascript
const formData = new FormData()
formData.append('file', imageFile)

const response = await fetch('http://127.0.0.1:8000/predict', {
  method: 'POST',
  body: formData
})
```

### Response Structure
```json
{
  "is_skin": { "label": "skin", "confidence": 0.95, "probabilities": {...} },
  "lesion_type": { "label_index": 1, "confidence": 0.87, "probabilities": [...] },
  "malignancy": { "label": "benign", "confidence": 0.92, "probabilities": {...} },
  "route_taken": ["SAM_features_real", "MLP1_loaded", "MLP2_loaded", "MLP3_loaded"],
  "metadata": { "image_name": "ISIC_0034321", "diagnosis_from_csv": "NV", ... }
}
```

## 🚀 Build & Deploy

### Development
```bash
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
```

### Production Build
```bash
npm run build
```

The built files will be in the `dist/` directory, ready for deployment.

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure backend is running on port 8000
   - Check CORS settings in backend
   - Verify network connectivity

2. **Image Upload Fails**
   - Check file size (max 10MB)
   - Verify file format is supported
   - Check browser console for errors

3. **No Metadata Displayed**
   - Image must be in the dataset
   - Filename must match CSV entries
   - Check backend metadata loading

### Debug Mode

Enable browser developer tools to see:
- Network requests
- Console errors
- Component state changes

## 🎯 Future Enhancements

- **Batch Processing**: Upload multiple images
- **Result History**: Save previous predictions
- **Export Results**: Download predictions as CSV
- **Advanced Filtering**: Filter by diagnosis type
- **User Authentication**: Secure access control

## 📄 License

This project is part of the Skin Lesion Classification research system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Note**: This frontend is designed to work with the specific backend API. Ensure backend compatibility before making changes.
