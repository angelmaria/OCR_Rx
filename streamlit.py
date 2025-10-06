import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import json
import sqlite3
from datetime import datetime
import pandas as pd
import io
import fitz  # PyMuPDF
from pathlib import Path
import os
import platform

# ============================================================================
# CONFIGURACIÓN TESSERACT
# ============================================================================

def configurar_tesseract():
    """Configura automáticamente la ruta de Tesseract según el sistema operativo"""
    
    # Si ya está configurado manualmente, respetarlo
    if pytesseract.pytesseract.tesseract_cmd and os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        return True
    
    sistema = platform.system()
    
    if sistema == "Windows":
        # Rutas comunes en Windows (ORDEN IMPORTANTE: específicas primero)
        posibles_rutas = [
            # TU RUTA ESPECÍFICA
            r"C:\Users\angel.martinez\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
            # Rutas estándar
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                pytesseract.pytesseract.tesseract_cmd = ruta
                print(f"✅ Tesseract encontrado en: {ruta}")
                return True
        
        print("❌ Tesseract no encontrado en rutas predefinidas")
        return False
    
    elif sistema == "Darwin":  # macOS
        # En Mac con Homebrew, tesseract suele estar en PATH
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                ruta = result.stdout.strip()
                pytesseract.pytesseract.tesseract_cmd = ruta
                print(f"✅ Tesseract encontrado en: {ruta}")
                return True
        except:
            pass
        
        # Rutas comunes en Mac
        posibles_rutas = [
            "/opt/homebrew/bin/tesseract",  # Apple Silicon (M1/M2/M3)
            "/usr/local/bin/tesseract",     # Intel Mac
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                pytesseract.pytesseract.tesseract_cmd = ruta
                print(f"✅ Tesseract encontrado en: {ruta}")
                return True
        
        print("❌ Tesseract no encontrado. Instala con: brew install tesseract")
        return False
    
    else:  # Linux
        # En Linux suele estar en PATH
        try:
            import subprocess
            result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except:
            pass
        return True

# Intentar configurar Tesseract al inicio
TESSERACT_DISPONIBLE = configurar_tesseract()

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Configurar página
st.set_page_config(
    page_title="OCR Fórmulas Magistrales",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BASE DE DATOS - SISTEMA DE APRENDIZAJE
# ============================================================================

class LearningDatabase:
    """Base de datos SQLite para almacenar correcciones y aprendizaje"""
    
    def __init__(self, db_path="medical_ocr_learning.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa las tablas si no existen"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla principal de recetas procesadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recetas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha_proceso TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tipo_receta TEXT,
                ocr_engine TEXT,
                texto_original TEXT,
                datos_extraidos TEXT,
                validado BOOLEAN DEFAULT 0,
                confianza REAL
            )
        """)
        
        # Tabla de correcciones humanas (el "aprendizaje")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correcciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receta_id INTEGER,
                campo TEXT,
                valor_ocr TEXT,
                valor_correcto TEXT,
                fecha_correccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (receta_id) REFERENCES recetas(id)
            )
        """)
        
        # Diccionario médico personalizado
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diccionario_medico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                termino TEXT UNIQUE,
                tipo TEXT,
                frecuencia INTEGER DEFAULT 1,
                ultima_vez TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Patrones aprendidos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patrones_aprendidos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patron_texto TEXT,
                campo_destino TEXT,
                exito_rate REAL,
                veces_usado INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def guardar_receta(self, tipo_receta, ocr_engine, texto_original, datos_extraidos, confianza):
        """Guarda una receta procesada"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO recetas (tipo_receta, ocr_engine, texto_original, datos_extraidos, confianza)
            VALUES (?, ?, ?, ?, ?)
        """, (tipo_receta, ocr_engine, texto_original, json.dumps(datos_extraidos), confianza))
        
        receta_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return receta_id
    
    def guardar_correccion(self, receta_id, campo, valor_ocr, valor_correcto):
        """Guarda una corrección humana (aprendizaje)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO correcciones (receta_id, campo, valor_ocr, valor_correcto)
            VALUES (?, ?, ?, ?)
        """, (receta_id, campo, valor_ocr, valor_correcto))
        
        # Actualizar diccionario si es un término médico
        if campo in ['medicamento', 'principio_activo']:
            cursor.execute("""
                INSERT INTO diccionario_medico (termino, tipo, frecuencia)
                VALUES (?, ?, 1)
                ON CONFLICT(termino) DO UPDATE SET 
                    frecuencia = frecuencia + 1,
                    ultima_vez = CURRENT_TIMESTAMP
            """, (valor_correcto, campo))
        
        conn.commit()
        conn.close()
    
    def buscar_correcciones_similares(self, valor_ocr, campo):
        """Busca correcciones previas similares (aprendizaje automático)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buscar correcciones exactas primero
        cursor.execute("""
            SELECT valor_correcto, COUNT(*) as veces
            FROM correcciones
            WHERE campo = ? AND LOWER(valor_ocr) = LOWER(?)
            GROUP BY valor_correcto
            ORDER BY veces DESC
            LIMIT 1
        """, (campo, valor_ocr))
        
        resultado = cursor.fetchone()
        conn.close()
        
        if resultado:
            return resultado[0], resultado[1]
        return None, 0
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas del sistema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM recetas")
        total_recetas = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM recetas WHERE validado = 1")
        validadas = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM correcciones")
        correcciones = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confianza) FROM recetas WHERE confianza IS NOT NULL")
        confianza_avg = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_recetas': total_recetas,
            'validadas': validadas,
            'correcciones': correcciones,
            'confianza_promedio': round(confianza_avg * 100, 1)
        }

# ============================================================================
# MOTORES OCR
# ============================================================================

class OCREngine:
    """Clase base para motores OCR"""
    
    @staticmethod
    def detect_document_type(image):
        """
        Detecta si la receta es digital o manuscrita
        Basado en análisis de bordes y contraste
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calcular nitidez (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calcular contraste
        contrast = gray.std()
        
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Heurística: documentos digitales tienen más nitidez y contraste
        if laplacian_var > 100 and contrast > 50:
            return "digital", 0.85
        elif edge_density < 0.1:
            return "manuscrita", 0.65
        else:
            return "mixta", 0.70
    
    @staticmethod
    def preprocess_image(image, doc_type):
        """Preprocesa imagen según tipo de documento"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if doc_type == "digital":
            # Para digitales: simple threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        else:
            # Para manuscritas: procesamiento más agresivo
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            return binary

class TesseractOCR(OCREngine):
    """Motor Tesseract - Mejor para recetas digitales"""
    
    @staticmethod
    def process(image, doc_type):
        # Verificar si Tesseract está disponible
        if not TESSERACT_DISPONIBLE:
            raise Exception(
                "Tesseract no está instalado o no se encuentra en el sistema.\n\n"
                "Soluciones:\n"
                "- Windows: Descarga desde https://github.com/UB-Mannheim/tesseract/wiki\n"
                "- Mac: brew install tesseract\n"
                "- Linux: sudo apt-get install tesseract-ocr\n\n"
                "O usa EasyOCR en su lugar (selecciona en el menú lateral)."
            )
        
        processed = OCREngine.preprocess_image(image, doc_type)
        
        # Configuración optimizada para español médico
        custom_config = r'--oem 3 --psm 6 -l spa'
        
        try:
            text = pytesseract.image_to_string(processed, config=custom_config)
        except pytesseract.pytesseract.TesseractNotFoundError:
            raise Exception("Error al ejecutar Tesseract. Verifica la instalación.")
        
        # Calcular confianza aproximada
        try:
            data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) / 100 if confidences else 0.5
        except:
            avg_confidence = 0.5
        
        return text, avg_confidence

class EasyOCR_Engine(OCREngine):
    """Motor EasyOCR - Mejor para manuscritas (requiere instalación)"""
    
    @staticmethod
    def process(image, doc_type):
        try:
            import easyocr
            reader = easyocr.Reader(['es'], gpu=False)
            
            processed = OCREngine.preprocess_image(image, doc_type)
            results = reader.readtext(processed, detail=1)
            
            # Extraer texto y confianza
            text = '\n'.join([result[1] for result in results])
            avg_confidence = np.mean([result[2] for result in results]) if results else 0.5
            
            return text, avg_confidence
        except ImportError:
            st.error("⚠️ EasyOCR no está instalado. Usa: pip install easyocr")
            return "", 0.0

class AzureVisionOCR(OCREngine):
    """Motor Azure Vision - Premium (requiere API key)"""
    
    @staticmethod
    def process(image, doc_type, api_key=None, endpoint=None):
        if not api_key or not endpoint:
            st.warning("⚠️ Azure Vision requiere API Key y Endpoint")
            return "", 0.0
        
        try:
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
            from msrest.authentication import CognitiveServicesCredentials
            import time
            
            # Cliente Azure
            client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))
            
            # Convertir imagen a bytes
            _, buffer = cv2.imencode('.png', image)
            image_stream = io.BytesIO(buffer)
            
            # Llamar a Azure Read API
            read_operation = client.read_in_stream(image_stream, raw=True)
            operation_id = read_operation.headers['Operation-Location'].split('/')[-1]
            
            # Esperar resultado
            while True:
                result = client.get_read_result(operation_id)
                if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                    break
                time.sleep(1)
            
            # Extraer texto
            text = ""
            confidence_sum = 0
            count = 0
            
            if result.status == OperationStatusCodes.succeeded:
                for page in result.analyze_result.read_results:
                    for line in page.lines:
                        text += line.text + '\n'
                        if hasattr(line, 'confidence'):
                            confidence_sum += line.confidence
                            count += 1
            
            avg_confidence = confidence_sum / count if count > 0 else 0.8
            
            return text, avg_confidence
            
        except Exception as e:
            st.error(f"❌ Error Azure Vision: {str(e)}")
            return "", 0.0

# ============================================================================
# EXTRACTOR DE DATOS ESTRUCTURADOS
# ============================================================================

class DataExtractor:
    """Extrae y normaliza datos de recetas médicas"""
    
    def __init__(self, db: LearningDatabase):
        self.db = db
        self.setup_patterns()
    
    def setup_patterns(self):
        """Patrones regex para extracción"""
        self.patterns = {
            # Medicamentos y principios activos
            'minoxidil': r'[Mm]inoxidil\s*(\d+(?:[,\.]\d+)?)\s*(?:mg|%)',
            'finasteride': r'[Ff]inasteride\s*(\d+(?:[,\.]\d+)?)\s*mg',
            'dutasteride': r'[Dd]utasteride\s*(\d+(?:[,\.]\d+)?)\s*mg',
            'biotina': r'[Bb]iotina\s*(\d+(?:[,\.]\d+)?)\s*(?:mg|mcg)',
            'latanoprost': r'[Ll]atanoprost\s*(\d+(?:[,\.]\d+)?)\s*%',
            
            # Forma farmacéutica
            'capsulas': r'(?:cápsulas?|capsulas?)\s*n?[=º°]?\s*(\d+)',
            'solucion': r'(?:solución|solucion)\s*(\d+)\s*ml',
            'crema': r'crema\s*(\d+)\s*(?:g|gr|gramos)',
            'gel': r'gel\s*(\d+)\s*(?:g|gr|gramos)',
            
            # Cantidades y dosis
            'cantidad_mg': r'(\d+(?:[,\.]\d+)?)\s*(?:mg|miligramos?)',
            'cantidad_ml': r'(\d+(?:[,\.]\d+)?)\s*ml',
            'cantidad_g': r'(\d+(?:[,\.]\d+)?)\s*(?:g|gr|gramos)',
            
            # Posología
            'tomar': r'tomar\s*(\d+)',
            'cada_horas': r'cada\s*(\d+)\s*(?:horas?|h)',
            'cada_dias': r'cada\s*(\d+)\s*(?:días?|d)',
            'veces_dia': r'(\d+)\s*(?:veces?|vez)\s*(?:al|por)\s*día',
            
            # Duración tratamiento
            'duracion': r'(?:durante|por)\s*(\d+)\s*(?:días?|meses?|semanas?)',
            
            # Datos administrativos
            'dni': r'(?:DNI|NIE)[:\s]*([0-9]{8}[A-Z])',
            'fecha': r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            'num_colegiado': r'(?:N[ºo°]?\s*Col(?:egiado)?|Núm\.\s*Col)[:\s]*([0-9]{8,10})',
            'num_receta': r'(?:Receta|Rec\.|N[ºo°])[:\s]*([A-Z0-9\-]+)',
        }
    
    def extract(self, text, usar_aprendizaje=True):
        """Extrae datos estructurados del texto"""
        datos = {
            'medicamentos': [],
            'dosis': [],
            'forma_farmaceutica': None,
            'cantidad_total': None,
            'posologia': {},
            'duracion': None,
            'datos_admin': {},
            'texto_completo': text,
            'confianza_extraccion': 0.0
        }
        
        text_lower = text.lower()
        campos_encontrados = 0
        total_campos = 8
        
        # Extraer medicamentos
        for med_name, pattern in [
            ('minoxidil', self.patterns['minoxidil']),
            ('finasteride', self.patterns['finasteride']),
            ('dutasteride', self.patterns['dutasteride']),
            ('biotina', self.patterns['biotina']),
            ('latanoprost', self.patterns['latanoprost'])
        ]:
            match = re.search(pattern, text_lower)
            if match:
                dosis = match.group(1).replace(',', '.')
                medicamento = {
                    'nombre': med_name.title(),
                    'dosis': f"{dosis}mg" if med_name != 'latanoprost' else f"{dosis}%"
                }
                
                # APRENDIZAJE: Buscar correcciones previas
                if usar_aprendizaje:
                    correccion, veces = self.db.buscar_correcciones_similares(med_name, 'medicamento')
                    if correccion and veces > 2:
                        medicamento['nombre'] = correccion
                        medicamento['aprendido'] = True
                
                datos['medicamentos'].append(medicamento)
                campos_encontrados += 1
        
        # Forma farmacéutica
        for forma, pattern in [('cápsulas', self.patterns['capsulas']),
                              ('solución', self.patterns['solucion']),
                              ('crema', self.patterns['crema']),
                              ('gel', self.patterns['gel'])]:
            match = re.search(pattern, text_lower)
            if match:
                datos['forma_farmaceutica'] = forma
                datos['cantidad_total'] = match.group(1)
                campos_encontrados += 1
                break
        
        # Posología
        match = re.search(self.patterns['tomar'], text_lower)
        if match:
            datos['posologia']['cantidad'] = match.group(1)
        
        match = re.search(self.patterns['cada_horas'], text_lower)
        if match:
            datos['posologia']['frecuencia_horas'] = match.group(1)
            campos_encontrados += 1
        
        match = re.search(self.patterns['veces_dia'], text_lower)
        if match:
            datos['posologia']['veces_al_dia'] = match.group(1)
            campos_encontrados += 1
        
        # Duración
        match = re.search(self.patterns['duracion'], text_lower)
        if match:
            datos['duracion'] = match.group(0)
            campos_encontrados += 1
        
        # Datos administrativos
        for campo, pattern in [('dni', self.patterns['dni']),
                              ('fecha', self.patterns['fecha']),
                              ('num_colegiado', self.patterns['num_colegiado']),
                              ('num_receta', self.patterns['num_receta'])]:
            match = re.search(pattern, text)
            if match:
                datos['datos_admin'][campo] = match.group(1)
                campos_encontrados += 1
        
        # Calcular confianza de extracción
        datos['confianza_extraccion'] = campos_encontrados / total_campos
        
        return datos

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def extract_text_from_pdf(pdf_file):
    """Extrae texto de PDF (si tiene texto seleccionable)"""
    try:
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        
        if len(text.strip()) > 50:  # Si tiene texto suficiente
            return text, 0.95, True  # Alta confianza
        
        return None, 0, False  # PDF es imagen, necesita OCR
    except:
        return None, 0, False

def convert_pdf_to_images(pdf_file):
    """Convierte PDF a imágenes para OCR"""
    pdf_bytes = pdf_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    images = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom para mejor OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    
    return images

def export_to_json(datos, filename="receta_extraida.json"):
    """Exporta datos a JSON"""
    json_str = json.dumps(datos, ensure_ascii=False, indent=2)
    return json_str

def export_to_excel(datos, filename="receta_extraida.xlsx"):
    """Exporta datos a Excel"""
    # Aplanar datos para DataFrame
    flat_data = []
    
    # Medicamentos
    for i, med in enumerate(datos.get('medicamentos', [])):
        flat_data.append({
            'Campo': f'Medicamento_{i+1}',
            'Valor': med.get('nombre', ''),
            'Detalle': med.get('dosis', '')
        })
    
    # Forma farmacéutica
    if datos.get('forma_farmaceutica'):
        flat_data.append({
            'Campo': 'Forma Farmacéutica',
            'Valor': datos['forma_farmaceutica'],
            'Detalle': datos.get('cantidad_total', '')
        })
    
    # Posología
    for key, value in datos.get('posologia', {}).items():
        flat_data.append({
            'Campo': f'Posología_{key}',
            'Valor': value,
            'Detalle': ''
        })
    
    # Datos administrativos
    for key, value in datos.get('datos_admin', {}).items():
        flat_data.append({
            'Campo': key,
            'Valor': value,
            'Detalle': ''
        })
    
    df = pd.DataFrame(flat_data)
    
    # Convertir a Excel en memoria
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Receta')
    
    return output.getvalue()

# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

def main():
    # Inicializar base de datos
    if 'db' not in st.session_state:
        st.session_state.db = LearningDatabase()
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = DataExtractor(st.session_state.db)
    
    # Inicializar diccionario farmacéutico
    if 'pharma_dict' not in st.session_state:
        from pharma_dictionary import PharmaDictionary
        st.session_state.pharma_dict = PharmaDictionary()
    
    # Header
    st.markdown('<p class="main-header">🏥 Sistema OCR - Fórmulas Magistrales</p>', unsafe_allow_html=True)
    st.markdown("**Sistema inteligente con aprendizaje continuo para normalización de recetas**")
    
    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # ADVERTENCIA: Tesseract no disponible
        if not TESSERACT_DISPONIBLE:
            st.error("⚠️ **Tesseract no encontrado**")
            
            sistema = platform.system()
            if sistema == "Windows":
                st.markdown("""
                **Para Windows:**
                1. [Descargar instalador](https://github.com/UB-Mannheim/tesseract/wiki)
                2. Instalar en la ruta por defecto
                3. Reiniciar Streamlit
                
                **O usa EasyOCR** (más abajo)
                """)
            elif sistema == "Darwin":
                st.markdown("""
                **Para Mac:**
                ```bash
                brew install tesseract
                brew install tesseract-lang
                ```
                Luego reinicia Streamlit
                """)
            
            st.markdown("---")
        else:
            st.success(f"✅ Tesseract disponible")
            if pytesseract.pytesseract.tesseract_cmd:
                st.caption(f"📍 Ruta: {pytesseract.pytesseract.tesseract_cmd}")
        
        # Selector de motor OCR
        st.subheader("Motor OCR")
        
        # Filtrar opciones según disponibilidad de Tesseract
        opciones_disponibles = ["EasyOCR (manuscrita)"]
        
        if TESSERACT_DISPONIBLE:
            opciones_disponibles = [
                "Automático (recomendado)", 
                "Tesseract (digital)", 
                "EasyOCR (manuscrita)", 
                "Azure Vision (premium)"
            ]
        else:
            opciones_disponibles.append("Azure Vision (premium)")
        
        ocr_engine = st.radio(
            "Selecciona motor:",
            opciones_disponibles,
            help="El modo automático detecta el tipo de documento y elige el mejor motor"
        )
        
        # Información de cada motor
        if "Automático" in ocr_engine:
            st.info("✅ **Automático**: Detecta tipo de documento y elige:\n- Tesseract para digitales\n- EasyOCR para manuscritas")
        elif "Tesseract" in ocr_engine:
            st.info("⚡ **Tesseract**: Rápido y gratis. Ideal para recetas informatizadas con texto claro.")
        elif "EasyOCR" in ocr_engine:
            st.warning("🧠 **EasyOCR**: Deep learning. Mejor para manuscritas. Más lento.\n\n`pip install easyocr`")
        elif "Azure" in ocr_engine:
            st.warning("☁️ **Azure Vision**: Premium. Mejor precisión.\n\nRequiere API Key")
            azure_key = st.text_input("Azure API Key", type="password")
            azure_endpoint = st.text_input("Azure Endpoint")
        
        st.markdown("---")
        
        # Estadísticas del sistema
        st.subheader("📊 Estadísticas")
        stats = st.session_state.db.obtener_estadisticas()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recetas", stats['total_recetas'])
            st.metric("Validadas", stats['validadas'])
        with col2:
            st.metric("Correcciones", stats['correcciones'])
            st.metric("Precisión", f"{stats['confianza_promedio']}%")
        
        if stats['correcciones'] > 0:
            st.success(f"🧠 Sistema aprendido de **{stats['correcciones']}** correcciones")
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📤 Procesar Receta", "✅ Validar y Corregir", "📈 Historial"])
    
    # ========================================================================
    # TAB 1: PROCESAR RECETA
    # ========================================================================
    with tab1:
        st.header("1. Cargar Receta")
        
        uploaded_file = st.file_uploader(
            "Sube una imagen o PDF de la receta",
            type=['png', 'jpg', 'jpeg', 'pdf', 'tiff'],
            help="Formatos soportados: PNG, JPG, PDF"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 Documento Original")
                
                # Determinar si es PDF o imagen
                is_pdf = uploaded_file.name.lower().endswith('.pdf')
                
                if is_pdf:
                    # Intentar extraer texto directamente del PDF
                    text_from_pdf, pdf_confidence, has_text = extract_text_from_pdf(uploaded_file)
                    
                    if has_text:
                        st.success("✅ PDF con texto seleccionable detectado")
                        st.session_state.extracted_text = text_from_pdf
                        st.session_state.ocr_confidence = pdf_confidence
                        st.session_state.doc_type = "digital"
                        
                        with st.expander("Ver texto extraído"):
                            st.text_area("Texto", text_from_pdf, height=300)
                    else:
                        st.warning("⚠️ PDF es imagen escaneada, requiere OCR")
                        # Convertir PDF a imágenes
                        uploaded_file.seek(0)
                        images = convert_pdf_to_images(uploaded_file)
                        st.image(images[0], caption="Página 1", use_container_width=True)
                        
                        # Guardar en session state para procesamiento
                        st.session_state.current_image = cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR)
                else:
                    # Es una imagen
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Receta cargada", use_container_width=True)
                    
                    # Convertir a formato OpenCV
                    image_np = np.array(image)
                    if len(image_np.shape) == 2:  # Grayscale
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                    elif image_np.shape[2] == 4:  # RGBA
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
                    else:  # RGB
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    st.session_state.current_image = image_np
            
            with col2:
                st.subheader("🔍 Análisis y Procesamiento")
                
                if st.button("🚀 Procesar con OCR", type="primary", use_container_width=True):
                    
                    if 'current_image' not in st.session_state and not has_text:
                        st.error("No hay imagen para procesar")
                    else:
                        with st.spinner("Procesando..."):
                            
                            # Si ya tenemos texto del PDF, saltamos OCR
                            if 'extracted_text' in st.session_state and has_text:
                                extracted_text = st.session_state.extracted_text
                                confidence = st.session_state.ocr_confidence
                                doc_type = "digital"
                                engine_used = "PDF directo"
                            else:
                                # Detectar tipo de documento
                                image = st.session_state.current_image
                                doc_type, type_confidence = OCREngine.detect_document_type(image)
                                
                                st.info(f"📋 Tipo detectado: **{doc_type.upper()}** (confianza: {type_confidence*100:.1f}%)")
                                
                                # Seleccionar motor OCR
                                if "Automático" in ocr_engine:
                                    if doc_type == "digital":
                                        if TESSERACT_DISPONIBLE:
                                            engine_used = "Tesseract"
                                            extracted_text, confidence = TesseractOCR.process(image, doc_type)
                                        else:
                                            engine_used = "EasyOCR (Tesseract no disponible)"
                                            extracted_text, confidence = EasyOCR_Engine.process(image, doc_type)
                                    else:
                                        engine_used = "EasyOCR"
                                        extracted_text, confidence = EasyOCR_Engine.process(image, doc_type)
                                
                                elif "Tesseract" in ocr_engine:
                                    if not TESSERACT_DISPONIBLE:
                                        st.error("❌ Tesseract no está disponible. Usa EasyOCR.")
                                        return
                                    engine_used = "Tesseract"
                                    extracted_text, confidence = TesseractOCR.process(image, doc_type)
                                
                                elif "EasyOCR" in ocr_engine:
                                    engine_used = "EasyOCR"
                                    extracted_text, confidence = EasyOCR_Engine.process(image, doc_type)
                                
                                elif "Azure" in ocr_engine:
                                    if azure_key and azure_endpoint:
                                        engine_used = "Azure Vision"
                                        extracted_text, confidence = AzureVisionOCR.process(
                                            image, doc_type, azure_key, azure_endpoint
                                        )
                                    else:
                                        st.error("❌ Falta configuración de Azure")
                                        return
                                
                                st.session_state.extracted_text = extracted_text
                                st.session_state.ocr_confidence = confidence
                                st.session_state.doc_type = doc_type
                            
                            # Mostrar resultado OCR
                            st.success(f"✅ OCR completado con **{engine_used}**")
                            st.metric("Confianza OCR", f"{confidence*100:.1f}%")
                            
                            with st.expander("📝 Ver texto extraído", expanded=True):
                                st.text_area("Texto OCR", extracted_text, height=250, key="ocr_text_display")
                            
                            # NOVEDAD: Expandir abreviaturas automáticamente
                            st.markdown("---")
                            st.subheader("📖 Abreviaturas Detectadas")
                            
                            texto_expandido, abreviaturas_encontradas = st.session_state.pharma_dict.expandir_abreviaturas(extracted_text)
                            
                            if abreviaturas_encontradas:
                                st.success(f"✅ {len(abreviaturas_encontradas)} abreviaturas detectadas")
                                
                                for abrev in abreviaturas_encontradas:
                                    with st.expander(f"**{abrev['abreviatura']}** → {abrev['nombre_completo']}", expanded=False):
                                        st.markdown(f"""
                                        **Categoría:** {abrev['categoria'].replace('_', ' ').title()}  
                                        **Descripción:** {abrev['descripcion']}
                                        """)
                                
                                # Guardar texto expandido
                                st.session_state.texto_expandido = texto_expandido
                            else:
                                st.info("ℹ️ No se detectaron abreviaturas comunes")
                            
                            # Extraer datos estructurados
                            st.markdown("---")
                            st.subheader("📊 Extrayendo datos estructurados...")
                            
                            datos_extraidos = st.session_state.extractor.extract(extracted_text)
                            st.session_state.datos_extraidos = datos_extraidos
                            
                            # Guardar en base de datos
                            receta_id = st.session_state.db.guardar_receta(
                                tipo_receta=doc_type,
                                ocr_engine=engine_used,
                                texto_original=extracted_text,
                                datos_extraidos=datos_extraidos,
                                confianza=confidence
                            )
                            st.session_state.current_receta_id = receta_id
                            
                            st.success(f"💾 Receta guardada (ID: {receta_id})")
        
        # Mostrar datos extraídos si existen
        if 'datos_extraidos' in st.session_state:
            st.markdown("---")
            st.header("2. Datos Normalizados")
            
            datos = st.session_state.datos_extraidos
            
            # Confianza general
            confianza_ext = datos.get('confianza_extraccion', 0)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Campos encontrados", len([k for k, v in datos.items() if v and k != 'texto_completo']))
            with col2:
                st.metric("Confianza extracción", f"{confianza_ext*100:.1f}%")
            with col3:
                if confianza_ext > 0.7:
                    st.success("✅ Alta confianza")
                elif confianza_ext > 0.4:
                    st.warning("⚠️ Confianza media")
                else:
                    st.error("❌ Baja confianza")
            
            # Medicamentos
            if datos.get('medicamentos'):
                st.subheader("💊 Medicamentos")
                for i, med in enumerate(datos['medicamentos']):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        aprendido = " 🧠" if med.get('aprendido') else ""
                        st.info(f"**{med['nombre']}** - Dosis: {med['dosis']}{aprendido}")
                    with col2:
                        if med.get('aprendido'):
                            st.caption("Corregido automáticamente")
                    with col3:
                        # NOVEDAD: Botón de info del medicamento
                        if st.button("ℹ️ Info", key=f"info_med_{i}"):
                            info_med = st.session_state.pharma_dict.obtener_info_principio_activo(med['nombre'])
                            if info_med:
                                st.session_state[f'show_info_{i}'] = True
                    
                    # Mostrar información adicional si se solicitó
                    if st.session_state.get(f'show_info_{i}'):
                        with st.container():
                            st.markdown(f"""
                            **📋 Información de {info_med['nombre_generico']}**
                            - **Categoría:** {info_med['categoria'].replace('_', ' ').title()}
                            - **Indicaciones:** {', '.join(info_med['indicaciones'])}
                            - **Dosis habituales:** {', '.join(info_med['dosis_habituales'])}
                            - **Formas farmacéuticas:** {', '.join(info_med['formas_farmaceuticas'])}
                            """)
                            if info_med['incompatibilidades']:
                                st.warning(f"⚠️ Incompatibilidades: {', '.join(info_med['incompatibilidades'])}")
                
                # NOVEDAD: Verificar incompatibilidades entre medicamentos
                st.markdown("---")
                st.subheader("🔍 Verificación de Compatibilidad")
                
                medicamentos_nombres = [med['nombre'] for med in datos['medicamentos']]
                incompatibilidades = []
                
                for med_nombre in medicamentos_nombres:
                    info = st.session_state.pharma_dict.obtener_info_principio_activo(med_nombre)
                    if info and info['incompatibilidades']:
                        for otro_med in medicamentos_nombres:
                            if otro_med != med_nombre and otro_med.lower() in [inc.lower() for inc in info['incompatibilidades']]:
                                incompatibilidades.append((med_nombre, otro_med))
                
                if incompatibilidades:
                    st.error("🚨 ADVERTENCIA: Incompatibilidades detectadas")
                    for med1, med2 in incompatibilidades:
                        st.warning(f"⚠️ {med1} + {med2}")
                else:
                    st.success("✅ No se detectaron incompatibilidades conocidas")
            
            # Forma farmacéutica
            if datos.get('forma_farmaceutica'):
                st.subheader("📦 Forma Farmacéutica")
                st.success(f"{datos['forma_farmaceutica'].title()} - Cantidad: {datos.get('cantidad_total', 'N/A')}")
            
            # Posología
            if datos.get('posologia'):
                st.subheader("⏰ Posología")
                pos = datos['posologia']
                posologia_text = []
                
                if pos.get('cantidad'):
                    posologia_text.append(f"Tomar {pos['cantidad']}")
                if pos.get('frecuencia_horas'):
                    posologia_text.append(f"cada {pos['frecuencia_horas']} horas")
                if pos.get('veces_al_dia'):
                    posologia_text.append(f"{pos['veces_al_dia']} veces al día")
                
                if posologia_text:
                    st.info(" ".join(posologia_text))
            
            # Duración
            if datos.get('duracion'):
                st.subheader("📅 Duración")
                st.info(datos['duracion'])
            
            # Datos administrativos
            if datos.get('datos_admin'):
                st.subheader("📋 Datos Administrativos")
                admin_df = pd.DataFrame([
                    {"Campo": k.replace('_', ' ').title(), "Valor": v}
                    for k, v in datos['datos_admin'].items()
                ])
                st.dataframe(admin_df, use_container_width=True, hide_index=True)
            
            # Botones de exportación
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                json_data = export_to_json(datos)
                st.download_button(
                    label="📥 Descargar JSON",
                    data=json_data,
                    file_name=f"receta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                excel_data = export_to_excel(datos)
                st.download_button(
                    label="📥 Descargar Excel",
                    data=excel_data,
                    file_name=f"receta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    # ========================================================================
    # TAB 2: VALIDAR Y CORREGIR (APRENDIZAJE)
    # ========================================================================
    with tab2:
        st.header("✅ Validar y Corregir - Sistema de Aprendizaje")
        
        st.markdown("""
        <div class="warning-box">
        <strong>🧠 Sistema de Aprendizaje Continuo</strong><br>
        Las correcciones que hagas aquí se guardarán en la base de datos y mejorarán 
        automáticamente las futuras extracciones de datos similares.
        </div>
        """, unsafe_allow_html=True)
        
        if 'datos_extraidos' not in st.session_state:
            st.info("👆 Primero procesa una receta en la pestaña anterior")
        else:
            datos = st.session_state.datos_extraidos
            receta_id = st.session_state.get('current_receta_id')
            
            st.subheader("Revisa y corrige los datos extraídos:")
            
            # Formulario de corrección
            with st.form("correction_form"):
                
                # Medicamentos
                st.markdown("### 💊 Medicamentos")
                medicamentos_corregidos = []
                
                if datos.get('medicamentos'):
                    for i, med in enumerate(datos['medicamentos']):
                        col1, col2 = st.columns(2)
                        with col1:
                            nombre = st.text_input(
                                f"Medicamento {i+1}",
                                value=med['nombre'],
                                key=f"med_nombre_{i}"
                            )
                        with col2:
                            dosis = st.text_input(
                                f"Dosis {i+1}",
                                value=med['dosis'],
                                key=f"med_dosis_{i}"
                            )
                        medicamentos_corregidos.append({'nombre': nombre, 'dosis': dosis})
                else:
                    st.info("No se detectaron medicamentos. Agrégalos manualmente:")
                    nombre = st.text_input("Medicamento", key="med_nombre_new")
                    dosis = st.text_input("Dosis", key="med_dosis_new")
                    if nombre:
                        medicamentos_corregidos.append({'nombre': nombre, 'dosis': dosis})
                
                # Forma farmacéutica
                st.markdown("### 📦 Forma Farmacéutica")
                col1, col2 = st.columns(2)
                with col1:
                    forma = st.selectbox(
                        "Forma",
                        ["Cápsulas", "Solución", "Crema", "Gel", "Comprimidos", "Otra"],
                        index=0 if not datos.get('forma_farmaceutica') else 
                              ["cápsulas", "solución", "crema", "gel"].index(datos['forma_farmaceutica']) 
                              if datos['forma_farmaceutica'] in ["cápsulas", "solución", "crema", "gel"] else 0
                    )
                with col2:
                    cantidad = st.text_input(
                        "Cantidad",
                        value=datos.get('cantidad_total', '')
                    )
                
                # Posología
                st.markdown("### ⏰ Posología")
                col1, col2, col3 = st.columns(3)
                with col1:
                    pos_cantidad = st.text_input(
                        "Tomar",
                        value=datos.get('posologia', {}).get('cantidad', '1')
                    )
                with col2:
                    pos_frecuencia = st.text_input(
                        "Cada (horas)",
                        value=datos.get('posologia', {}).get('frecuencia_horas', '')
                    )
                with col3:
                    pos_veces = st.text_input(
                        "Veces al día",
                        value=datos.get('posologia', {}).get('veces_al_dia', '')
                    )
                
                # Duración
                st.markdown("### 📅 Duración del Tratamiento")
                duracion = st.text_input(
                    "Duración",
                    value=datos.get('duracion', ''),
                    placeholder="Ej: 30 días, 3 meses"
                )
                
                # Datos administrativos
                st.markdown("### 📋 Datos Administrativos")
                col1, col2 = st.columns(2)
                with col1:
                    dni = st.text_input(
                        "DNI/NIE Paciente",
                        value=datos.get('datos_admin', {}).get('dni', '')
                    )
                    num_receta = st.text_input(
                        "Número de Receta",
                        value=datos.get('datos_admin', {}).get('num_receta', '')
                    )
                with col2:
                    fecha = st.text_input(
                        "Fecha",
                        value=datos.get('datos_admin', {}).get('fecha', '')
                    )
                    num_colegiado = st.text_input(
                        "Núm. Colegiado",
                        value=datos.get('datos_admin', {}).get('num_colegiado', '')
                    )
                
                # Botón de validar
                submitted = st.form_submit_button("✅ Validar y Guardar Correcciones", use_container_width=True, type="primary")
                
                if submitted:
                    # Guardar correcciones en la base de datos (APRENDIZAJE)
                    correcciones_guardadas = 0
                    
                    # Comparar y guardar medicamentos
                    for i, med_original in enumerate(datos.get('medicamentos', [])):
                        if i < len(medicamentos_corregidos):
                            med_corregido = medicamentos_corregidos[i]
                            
                            if med_original['nombre'] != med_corregido['nombre']:
                                st.session_state.db.guardar_correccion(
                                    receta_id,
                                    'medicamento',
                                    med_original['nombre'],
                                    med_corregido['nombre']
                                )
                                correcciones_guardadas += 1
                            
                            if med_original['dosis'] != med_corregido['dosis']:
                                st.session_state.db.guardar_correccion(
                                    receta_id,
                                    'dosis',
                                    med_original['dosis'],
                                    med_corregido['dosis']
                                )
                                correcciones_guardadas += 1
                    
                    # Marcar receta como validada
                    conn = sqlite3.connect(st.session_state.db.db_path)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE recetas SET validado = 1 WHERE id = ?", (receta_id,))
                    conn.commit()
                    conn.close()
                    
                    st.success(f"""
                    ✅ **Receta validada exitosamente**
                    
                    - {correcciones_guardadas} correcciones guardadas
                    - El sistema aprenderá de estas correcciones
                    - Las próximas recetas similares serán más precisas
                    """)
                    
                    st.balloons()
    
    # ========================================================================
    # TAB 3: HISTORIAL
    # ========================================================================
    with tab3:
        st.header("📈 Historial y Diccionario")
        
        # Sub-tabs
        subtab1, subtab2, subtab3 = st.tabs(["📋 Recetas", "📖 Diccionario", "📊 Estadísticas"])
        
        # SUBTAB 1: Historial de recetas (código existente)
        with subtab1:
            st.subheader("Historial de Recetas Procesadas")
            
            # Obtener todas las recetas de la BD
            conn = sqlite3.connect(st.session_state.db.db_path)
            
            query = """
            SELECT 
                id,
                fecha_proceso,
                tipo_receta,
                ocr_engine,
                validado,
                confianza
            FROM recetas
            ORDER BY fecha_proceso DESC
            LIMIT 50
            """
            
            df_recetas = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df_recetas) == 0:
                st.info("📭 No hay recetas procesadas aún")
            else:
                # Métricas generales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Recetas", len(df_recetas))
                with col2:
                    validadas = df_recetas['validado'].sum()
                    st.metric("Validadas", f"{validadas} ({validadas/len(df_recetas)*100:.0f}%)")
                with col3:
                    avg_conf = df_recetas['confianza'].mean() * 100
                    st.metric("Confianza Promedio", f"{avg_conf:.1f}%")
                with col4:
                    manuscritas = (df_recetas['tipo_receta'] == 'manuscrita').sum()
                    st.metric("Manuscritas", manuscritas)
                
                st.markdown("---")
                
                # Filtros
                col1, col2 = st.columns(2)
                with col1:
                    filtro_tipo = st.multiselect(
                        "Filtrar por tipo",
                        options=df_recetas['tipo_receta'].unique().tolist(),
                        default=df_recetas['tipo_receta'].unique().tolist()
                    )
                with col2:
                    filtro_validado = st.selectbox(
                        "Estado validación",
                        ["Todas", "Validadas", "Sin validar"]
                    )
                
                # Aplicar filtros
                df_filtrado = df_recetas[df_recetas['tipo_receta'].isin(filtro_tipo)]
                
                if filtro_validado == "Validadas":
                    df_filtrado = df_filtrado[df_filtrado['validado'] == 1]
                elif filtro_validado == "Sin validar":
                    df_filtrado = df_filtrado[df_filtrado['validado'] == 0]
                
                # Formatear para visualización
                df_display = df_filtrado.copy()
                df_display['fecha_proceso'] = pd.to_datetime(df_display['fecha_proceso']).dt.strftime('%Y-%m-%d %H:%M')
                df_display['validado'] = df_display['validado'].map({1: '✅', 0: '⏳'})
                df_display['confianza'] = (df_display['confianza'] * 100).round(1).astype(str) + '%'
                
                df_display.columns = ['ID', 'Fecha', 'Tipo', 'Motor OCR', 'Validado', 'Confianza']
                
                # Mostrar tabla
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Gráficos
                st.markdown("---")
                st.subheader("📊 Análisis Temporal")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribución por tipo
                    tipo_counts = df_recetas['tipo_receta'].value_counts()
                    st.bar_chart(tipo_counts, use_container_width=True)
                    st.caption("Distribución por tipo de receta")
                
                with col2:
                    # Evolución de confianza
                    df_recetas['fecha'] = pd.to_datetime(df_recetas['fecha_proceso']).dt.date
                    conf_by_date = df_recetas.groupby('fecha')['confianza'].mean() * 100
                    st.line_chart(conf_by_date, use_container_width=True)
                    st.caption("Evolución de confianza promedio")
        
        # SUBTAB 2: DICCIONARIO FARMACÉUTICO
        with subtab2:
            st.subheader("📖 Diccionario Farmacéutico")
            
            # Estadísticas del diccionario
            dict_stats = st.session_state.pharma_dict.obtener_estadisticas()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Abreviaturas", dict_stats['total_abreviaturas'])
            with col2:
                st.metric("Principios Activos", dict_stats['total_principios_activos'])
            with col3:
                st.metric("Variantes Ortográficas", dict_stats['total_variantes'])
            
            st.markdown("---")
            
            # Abreviaturas más usadas
            st.subheader("🔥 Términos Más Usados")
            
            if dict_stats['mas_usadas']:
                df_top = pd.DataFrame(dict_stats['mas_usadas'])
                
                # Mostrar en tabla bonita
                for idx, row in df_top.iterrows():
                    with st.expander(f"**{row['abrev']}** - {row['nombre']} ({row['usos']} usos)"):
                        # Buscar info completa
                        conn = sqlite3.connect(st.session_state.pharma_dict.db_path)
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT categoria, descripcion, uso_comun
                            FROM abreviaturas
                            WHERE abreviatura = ?
                        """, (row['abrev'],))
                        info = cursor.fetchone()
                        conn.close()
                        
                        if info:
                            st.markdown(f"""
                            **Categoría:** {info[0].replace('_', ' ').title()}  
                            **Descripción:** {info[1]}  
                            **Uso común:** {info[2]}
                            """)
            else:
                st.info("ℹ️ Aún no se han usado términos del diccionario")
            
            st.markdown("---")
            
            # Agregar término personalizado
            st.subheader("➕ Agregar Término Personalizado")
            
            with st.form("add_custom_term"):
                col1, col2 = st.columns(2)
                
                with col1:
                    nueva_abrev = st.text_input("Abreviatura *", placeholder="Ej: SOE")
                    categoria = st.selectbox("Categoría *", [
                        "excipiente",
                        "via_administracion", 
                        "frecuencia",
                        "forma_farmaceutica",
                        "instruccion",
                        "conservante",
                        "otro"
                    ])
                
                with col2:
                    nuevo_nombre = st.text_input("Nombre Completo *", placeholder="Ej: Sin Otra Especificación")
                    uso_comun = st.text_input("Uso Común", placeholder="Dónde se usa normalmente")
                
                descripcion = st.text_area("Descripción", placeholder="Explica qué significa y para qué sirve")
                
                submitted = st.form_submit_button("💾 Guardar en Diccionario", use_container_width=True)
                
                if submitted:
                    if nueva_abrev and nuevo_nombre:
                        st.session_state.pharma_dict.agregar_termino_personalizado(
                            nueva_abrev, nuevo_nombre, categoria, descripcion
                        )
                        st.success(f"✅ '{nueva_abrev}' agregado correctamente")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ La abreviatura y el nombre completo son obligatorios")
            
            st.markdown("---")
            
            # Buscador de términos
            st.subheader("🔍 Buscar en Diccionario")
            
            busqueda = st.text_input("Busca una abreviatura o término", placeholder="Ej: BID, minoxidil, CSP")
            
            if busqueda:
                conn = sqlite3.connect(st.session_state.pharma_dict.db_path)
                cursor = conn.cursor()
                
                # Buscar en abreviaturas
                cursor.execute("""
                    SELECT abreviatura, nombre_completo, categoria, descripcion
                    FROM abreviaturas
                    WHERE LOWER(abreviatura) LIKE ? OR LOWER(nombre_completo) LIKE ?
                    LIMIT 10
                """, (f"%{busqueda.lower()}%", f"%{busqueda.lower()}%"))
                
                resultados_abrev = cursor.fetchall()
                
                # Buscar en principios activos
                cursor.execute("""
                    SELECT nombre_generico, categoria, indicaciones
                    FROM principios_activos
                    WHERE LOWER(nombre_generico) LIKE ?
                    LIMIT 10
                """, (f"%{busqueda.lower()}%",))
                
                resultados_pa = cursor.fetchall()
                conn.close()
                
                if resultados_abrev:
                    st.success(f"✅ {len(resultados_abrev)} abreviaturas encontradas")
                    for abrev, nombre, cat, desc in resultados_abrev:
                        st.info(f"**{abrev}** → {nombre}")
                        st.caption(f"Categoría: {cat} | {desc}")
                        st.markdown("---")
                
                if resultados_pa:
                    st.success(f"✅ {len(resultados_pa)} principios activos encontrados")
                    for nombre, cat, ind_json in resultados_pa:
                        indicaciones = json.loads(ind_json) if ind_json else []
                        st.info(f"**{nombre}** ({cat})")
                        if indicaciones:
                            st.caption(f"Indicaciones: {', '.join(indicaciones)}")
                        st.markdown("---")
                
                if not resultados_abrev and not resultados_pa:
                    st.warning("⚠️ No se encontraron resultados")
        
        # SUBTAB 3: ESTADÍSTICAS GLOBALES
        with subtab3:
            st.subheader("📊 Estadísticas del Sistema")
            
            # Combinar stats de OCR y Diccionario
            ocr_stats = st.session_state.db.obtener_estadisticas()
            dict_stats = st.session_state.pharma_dict.obtener_estadisticas()
            
            # Métricas principales
            st.markdown("### 🎯 Rendimiento General")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Precisión OCR",
                    f"{ocr_stats['confianza_promedio']}%",
                    delta=None
                )
            
            with col2:
                tasa_validacion = (ocr_stats['validadas'] / ocr_stats['total_recetas'] * 100) if ocr_stats['total_recetas'] > 0 else 0
                st.metric(
                    "Tasa Validación",
                    f"{tasa_validacion:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Correcciones Aprendidas",
                    ocr_stats['correcciones']
                )
            
            with col4:
                st.metric(
                    "Términos en Diccionario",
                    dict_stats['total_abreviaturas']
                )
            
            st.markdown("---")
            
            # Impacto del aprendizaje
            st.markdown("### 🧠 Impacto del Aprendizaje")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Sistema de Correcciones:**
                - Base de datos SQLite local
                - Persistencia entre sesiones
                - Mejora automática con uso
                """)
                
                if ocr_stats['correcciones'] > 0:
                    st.success(f"✅ {ocr_stats['correcciones']} correcciones guardadas")
                    st.caption("El sistema aprende de cada validación manual")
                else:
                    st.info("ℹ️ Aún no hay correcciones. Valida recetas para entrenar el sistema.")
            
            with col2:
                st.markdown("""
                **Diccionario Farmacéutico:**
                - Expansión automática de abreviaturas
                - Detección de incompatibilidades
                - Términos personalizables
                """)
                
                if dict_stats['mas_usadas']:
                    st.success(f"✅ {len(dict_stats['mas_usadas'])} términos activos")
                    st.caption("Términos más usados disponibles en pestaña anterior")
                else:
                    st.info("ℹ️ Procesa recetas para activar el diccionario")
            
            st.markdown("---")
            
            # Recomendaciones
            st.markdown("### 💡 Recomendaciones")
            
            if ocr_stats['total_recetas'] < 10:
                st.warning("⚠️ Procesa al menos 10 recetas para obtener estadísticas significativas")
            
            if ocr_stats['validadas'] < ocr_stats['total_recetas'] * 0.5:
                st.warning("⚠️ Valida más recetas para mejorar el aprendizaje del sistema")
            
            if dict_stats['total_abreviaturas'] < 30:
                st.info("💡 Agrega términos personalizados de tu farmacia al diccionario")

# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    main()