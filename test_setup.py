#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias están instaladas correctamente
"""

import sys
import os

def test_imports():
    """Verifica que todas las librerías se pueden importar"""
    print("Verificando imports...")
    
    try:
        import cv2
        print("✓ OpenCV (cv2) - OK")
        print(f"  Versión: {cv2.__version__}")
    except ImportError as e:
        print("✗ Error importing cv2:", e)
        return False
    
    try:
        import numpy as np
        print("✓ NumPy - OK")
        print(f"  Versión: {np.__version__}")
    except ImportError as e:
        print("✗ Error importing numpy:", e)
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow (PIL) - OK")
    except ImportError as e:
        print("✗ Error importing PIL:", e)
        return False
    
    try:
        import pytesseract
        print("✓ PyTesseract - OK")
    except ImportError as e:
        print("✗ Error importing pytesseract:", e)
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas - OK")
        print(f"  Versión: {pd.__version__}")
    except ImportError as e:
        print("✗ Error importing pandas:", e)
        return False
    
    try:
        import tkinter as tk
        print("✓ Tkinter - OK")
    except ImportError as e:
        print("✗ Error importing tkinter:", e)
        return False
    
    return True

def test_tesseract():
    """Verifica que Tesseract funciona correctamente"""
    print("\nVerificando Tesseract...")
    
    import pytesseract
    from PIL import Image
    import numpy as np
    
    # Configurar ruta de Tesseract
    tesseract_path = r'C:\Users\angel.martinez\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    
    if not os.path.exists(tesseract_path):
        print(f"✗ Tesseract no encontrado en: {tesseract_path}")
        print("Verifica la ruta de instalación.")
        return False
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✓ Tesseract encontrado en: {tesseract_path}")
    
    try:
        # Crear imagen de prueba
        import cv2
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, 'Texto de prueba', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Convertir a PIL
        pil_image = Image.fromarray(test_image)
        
        # Probar OCR
        text = pytesseract.image_to_string(pil_image, lang='spa')
        print("✓ OCR básico funciona")
        print(f"  Texto detectado: '{text.strip()}'")
        
        # Verificar idiomas disponibles
        langs = pytesseract.get_languages()
        if 'spa' in langs:
            print("✓ Idioma español (spa) disponible")
        else:
            print("⚠ Idioma español (spa) NO disponible")
            print("  Descarga spa.traineddata y colócalo en tessdata/")
        
        print(f"  Idiomas disponibles: {', '.join(langs)}")
        
    except Exception as e:
        print(f"✗ Error probando Tesseract: {e}")
        return False
    
    return True

def test_opencv():
    """Verifica funcionalidades básicas de OpenCV"""
    print("\nVerificando OpenCV...")
    
    try:
        import cv2
        import numpy as np
        
        # Crear imagen de prueba
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✓ Conversión de color funciona")
        
        # Probar filtros básicos
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        print("✓ Filtros básicos funcionan")
        
        return True
        
    except Exception as e:
        print(f"✗ Error probando OpenCV: {e}")
        return False

def main():
    print("=" * 50)
    print("VERIFICACIÓN DE DEPENDENCIAS OCR MÉDICO")
    print("=" * 50)
    
    success = True
    
    success &= test_imports()
    success &= test_tesseract()
    success &= test_opencv()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ TODAS LAS VERIFICACIONES PASARON")
        print("Puedes ejecutar la aplicación principal.")
    else:
        print("✗ ALGUNAS VERIFICACIONES FALLARON")
        print("Revisa los errores arriba y instala lo que falte.")
    print("=" * 50)

if __name__ == "__main__":
    main()