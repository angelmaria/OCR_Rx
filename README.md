# Medical Prescription OCR & Data Standardization

A Python application for extracting and standardizing data from medical prescription images using OCR (Optical Character Recognition) technology.

## Features

- **Image Processing**: Advanced image preprocessing for improved OCR accuracy
- **Multilingual OCR**: Supports Spanish medical prescriptions using Tesseract OCR
- **Intelligent Data Extraction**: Pattern-based extraction of key prescription information
- **Structured Output**: Organized data display with confidence levels
- **Export Options**: JSON and Excel export functionality
- **User-Friendly GUI**: Tkinter-based interface for easy operation

## Extracted Data Fields

The application automatically identifies and extracts:

- **Medications**: Minoxidil, Finasteride, Latanoprost (with dosages)
- **Quantities**: Number of capsules/units
- **Patient Information**: DNI/NIE identification numbers
- **Prescription Details**: Prescription numbers and codes
- **Prescriber Information**: Medical license numbers
- **Pharmacy Codes**: Pharmacy identification numbers
- **Dosage Instructions**: Basic posology information
- **Dates**: Multiple date fields from prescriptions

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Windows, macOS, or Linux
- Tesseract OCR engine

### Python Package Manager
This project supports both standard `pip` and `uv` package managers.

## Installation

### 1. Install Tesseract OCR

#### Windows
1. Download Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to default location or note custom path
3. Download Spanish language pack: [spa.traineddata](https://github.com/tesseract-ocr/tessdata/raw/main/spa.traineddata)
4. Place `spa.traineddata` in Tesseract's `tessdata` folder

#### macOS
```bash
brew install tesseract tesseract-lang
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-spa
```

### 2. Clone or Download Project

```bash
git clone <repository-url>
cd medical-prescription-ocr
```

### 3. Set Up Python Environment

#### Option A: Using UV (recommended)
```bash
# Create virtual environment
uv venv

# Activate environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip install opencv-python pillow pytesseract pandas openpyxl numpy
```

#### Option B: Using Standard PIP
```bash
# Create virtual environment
python -m venv ocr_env

# Activate environment
# Windows
ocr_env\Scripts\activate
# macOS/Linux
source ocr_env/bin/activate

# Install dependencies
pip install opencv-python pillow pytesseract pandas openpyxl numpy
```

### 4. Configure Tesseract Path

Edit the `main.py` file and update the Tesseract path:

```python
# Update this line with your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Path\To\Your\tesseract.exe'
```

Common paths:
- **Windows**: `C:\Users\{username}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe`
- **macOS**: `/usr/local/bin/tesseract` or `/opt/homebrew/bin/tesseract`
- **Linux**: `/usr/bin/tesseract`

## Usage

### 1. Verify Installation
Run the verification script to ensure all dependencies are properly installed:

```bash
python test_setup.py
```

### 2. Launch Application
```bash
python main.py
```

### 3. Processing Workflow

1. **Load Image**: Click "Cargar Imagen" to select a prescription image
2. **Process OCR**: Click "Procesar OCR" to extract text from the image
3. **Extract Data**: Click "Extraer Datos" to parse structured information
4. **Export Results**: Use "Exportar JSON" or "Exportar Excel" to save results

## Supported File Formats

- **Input Images**: PNG, JPG, JPEG, BMP, TIFF, GIF
- **Export Formats**: JSON, Excel (.xlsx)

## Configuration

### Image Preprocessing Parameters

The application includes several image preprocessing steps that can be adjusted in the `preprocess_image()` function:

- **Denoising**: Bilateral filter for noise reduction
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Binarization**: Adaptive thresholding for text clarity

### Pattern Customization

Regular expression patterns for data extraction can be modified in the `setup_patterns()` method to accommodate different prescription formats or additional fields.

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   uv pip install opencv-python
   # or
   pip install opencv-python
   ```

2. **"TesseractNotFoundError"**
   - Verify Tesseract installation
   - Update path in `main.py`
   - Ensure Spanish language pack is installed

3. **Poor OCR Accuracy**
   - Ensure image quality is sufficient
   - Check that Spanish language pack (`spa.traineddata`) is installed
   - Consider image preprocessing parameter adjustment

4. **Missing Spanish Language Support**
   - Download `spa.traineddata` from Tesseract repository
   - Place in Tesseract's `tessdata` directory

### Verification Commands

```bash
# Check Tesseract installation
tesseract --version

# List available languages
tesseract --list-langs

# Test Python environment
python -c "import cv2, pytesseract; print('All modules imported successfully')"
```

## Development

### Project Structure
```
medical-prescription-ocr/
├── main.py              # Main application
├── test_setup.py        # Installation verification
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── examples/           # Sample images (optional)
```

### Extending the Application

To add new data extraction patterns:

1. Add regex pattern to `setup_patterns()` method
2. Include extraction logic in `extract_data()` method
3. Update the data structure as needed

## Dependencies

- **opencv-python**: Image processing and computer vision
- **pillow**: Python Imaging Library
- **pytesseract**: Tesseract OCR wrapper
- **pandas**: Data manipulation and Excel export
- **openpyxl**: Excel file operations
- **numpy**: Numerical operations
- **tkinter**: GUI framework (included with Python)

## License

This project is provided as-is for educational and research purposes. Ensure compliance with local regulations regarding medical data processing.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Disclaimer

This tool is designed for administrative purposes and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always verify extracted data against original prescriptions.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify installation with `test_setup.py`
3. Review Tesseract OCR documentation
4. Open an issue with detailed error information