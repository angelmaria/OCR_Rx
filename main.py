import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import re
import json
from datetime import datetime
import pandas as pd

# Configurar la ruta de Tesseract para Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\angel.martinez\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class MedicalPrescriptionOCR:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Recetas Médicas")
        self.root.geometry("1200x800")
        
        # Variables para almacenar datos
        self.current_image = None
        self.extracted_text = ""
        self.standardized_data = {}
        
        self.setup_ui()
        self.setup_patterns()
        
    def setup_patterns(self):
        """Define patrones regex para extraer información específica"""
        self.patterns = {
            # Medicamentos comunes
            'minoxidil': r'[Mm]inoxidil\s*(\d+(?:[,\.]\d+)?)\s*(?:mg|miligramos?)',
            'finasteride': r'[Ff]inasteride\s*(\d+(?:[,\.]\d+)?)\s*(?:mg|miligramos?)',
            'latanoprost': r'[Ll]atanoprost\s*(\d+(?:[,\.]\d+)?)\s*%',
            
            # Dosis y cantidades
            'capsules': r'(?:cápsulas?|capsulas?)\s*n?[=º°]?\s*(\d+)',
            'dosage_general': r'(\d+(?:[,\.]\d+)?)\s*(?:mg|miligramos?|ml|g)',
            
            # Información del paciente (patrones generales)
            'dni_nie': r'(?:DNI|NIE|NIF)[:\s]*([0-9]{7,8}[A-Z])',
            'fecha': r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            
            # Números de receta y códigos
            'receta_number': r'(?:Receta|EF\s*N[ºo°]?)[:\s]*([A-Z0-9\-]+)',
            'codigo': r'(?:Código|Code)[:\s]*([A-Z0-9]+)',
            
            # Información del prescriptor
            'num_colegiado': r'(?:Núm|N[ºo°]?)\s*[Cc]olegiado[:\s]*([0-9]{8,9})',
            'doctor_name': r'Dr[a]?\.\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)',
            
            # Farmacia
            'farmacia_code': r'(?:Farmacia|SOE)[:\s]*([0-9]{4,6})',
            
            # Posología
            'posologia': r'(\d+)\s*(?:cada|c/)\s*(\d+)\s*(?:horas?|días?|h|d)',
            'duracion': r'(?:durante|por)\s*(\d+)\s*(?:días?|meses?|semanas?)'
        }
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Botones superiores
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(button_frame, text="Cargar Imagen", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Procesar OCR", command=self.process_ocr).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Extraer Datos", command=self.extract_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Exportar JSON", command=self.export_json).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Exportar Excel", command=self.export_excel).pack(side=tk.LEFT)
        
        # Frame para imagen
        image_frame = ttk.LabelFrame(main_frame, text="Imagen Original", padding="5")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.image_label = ttk.Label(image_frame, text="No hay imagen cargada")
        self.image_label.pack(expand=True)
        
        # Frame para texto OCR
        ocr_frame = ttk.LabelFrame(main_frame, text="Texto Extraído (OCR)", padding="5")
        ocr_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.text_ocr = tk.Text(ocr_frame, wrap=tk.WORD, height=10, width=50)
        scrollbar_ocr = ttk.Scrollbar(ocr_frame, orient=tk.VERTICAL, command=self.text_ocr.yview)
        self.text_ocr.configure(yscrollcommand=scrollbar_ocr.set)
        
        self.text_ocr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ocr.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame para datos estandarizados
        data_frame = ttk.LabelFrame(main_frame, text="Datos Estandarizados", padding="5")
        data_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Crear Treeview para mostrar datos estructurados
        columns = ('Campo', 'Valor', 'Confianza')
        self.tree = ttk.Treeview(data_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=200)
        
        scrollbar_tree = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar_tree.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_tree.pack(side=tk.RIGHT, fill=tk.Y)
        
    def load_image(self):
        """Carga una imagen desde archivo"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de receta",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            try:
                # Cargar y mostrar la imagen
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("No se pudo cargar la imagen")
                
                # Convertir para mostrar en Tkinter
                image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                # Redimensionar si es muy grande
                height, width = image_rgb.shape[:2]
                if width > 400 or height > 400:
                    scale = min(400/width, 400/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_rgb = cv2.resize(image_rgb, (new_width, new_height))
                
                image_pil = Image.fromarray(image_rgb)
                image_tk = ImageTk.PhotoImage(image_pil)
                
                self.image_label.configure(image=image_tk, text="")
                self.image_label.image = image_tk  # Mantener referencia
                
                messagebox.showinfo("Éxito", "Imagen cargada correctamente")
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
    
    def preprocess_image(self, image):
        """Preprocesa la imagen para mejorar el OCR"""
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Aplicar filtro bilateral para reducir ruido manteniendo bordes
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Aumentar contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Aplicar umbralización adaptativa
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def process_ocr(self):
        """Procesa la imagen con OCR"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        try:
            # Preprocesar imagen
            processed_image = self.preprocess_image(self.current_image)
            
            # Configurar Tesseract para español
            custom_config = r'--oem 3 --psm 6 -l spa'
            
            # Extraer texto
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                config=custom_config
            )
            
            self.extracted_text = extracted_text
            
            # Mostrar texto en el widget
            self.text_ocr.delete(1.0, tk.END)
            self.text_ocr.insert(1.0, extracted_text)
            
            messagebox.showinfo("Éxito", "OCR procesado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en OCR: {str(e)}")
    
    def extract_data(self):
        """Extrae datos estructurados del texto OCR"""
        if not self.extracted_text:
            messagebox.showwarning("Advertencia", "Primero procesa el OCR")
            return
        
        self.standardized_data = {}
        text = self.extracted_text.lower()
        original_text = self.extracted_text
        
        # Limpiar el Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Extraer información usando patrones
        extractions = []
        
        # Medicamento principal
        for med_name, pattern in [('minoxidil', self.patterns['minoxidil']), 
                                 ('finasteride', self.patterns['finasteride']),
                                 ('latanoprost', self.patterns['latanoprost'])]:
            match = re.search(pattern, text)
            if match:
                dose = match.group(1).replace(',', '.')
                self.standardized_data['medicamento'] = med_name.title()
                self.standardized_data['dosis'] = f"{dose}mg" if med_name != 'latanoprost' else f"{dose}%"
                extractions.append(('Medicamento', med_name.title(), 'Alta'))
                extractions.append(('Dosis', self.standardized_data['dosis'], 'Alta'))
                break
        
        # Número de cápsulas/unidades
        match = re.search(self.patterns['capsules'], text)
        if match:
            self.standardized_data['cantidad'] = match.group(1)
            extractions.append(('Cantidad', match.group(1), 'Media'))
        
        # DNI/NIE
        match = re.search(self.patterns['dni_nie'], original_text)
        if match:
            self.standardized_data['dni_paciente'] = match.group(1)
            extractions.append(('DNI/NIE Paciente', match.group(1), 'Alta'))
        
        # Fechas
        dates = re.findall(self.patterns['fecha'], original_text)
        if dates:
            self.standardized_data['fechas'] = dates
            for i, date in enumerate(dates[:3]):  # Máximo 3 fechas
                extractions.append((f'Fecha {i+1}', date, 'Media'))
        
        # Número de receta
        match = re.search(self.patterns['receta_number'], original_text)
        if match:
            self.standardized_data['numero_receta'] = match.group(1)
            extractions.append(('Número Receta', match.group(1), 'Alta'))
        
        # Número de colegiado
        match = re.search(self.patterns['num_colegiado'], original_text)
        if match:
            self.standardized_data['num_colegiado'] = match.group(1)
            extractions.append(('Núm. Colegiado', match.group(1), 'Alta'))
        
        # Código de farmacia
        match = re.search(self.patterns['farmacia_code'], original_text)
        if match:
            self.standardized_data['codigo_farmacia'] = match.group(1)
            extractions.append(('Código Farmacia', match.group(1), 'Media'))
        
        # Posología básica
        match = re.search(self.patterns['posologia'], text)
        if match:
            frecuencia = f"{match.group(1)} cada {match.group(2)} horas"
            self.standardized_data['posologia'] = frecuencia
            extractions.append(('Posología', frecuencia, 'Media'))
        
        # Agregar timestamp
        self.standardized_data['timestamp'] = datetime.now().isoformat()
        extractions.append(('Procesado', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Alta'))
        
        # Mostrar en Treeview
        for field, value, confidence in extractions:
            self.tree.insert('', tk.END, values=(field, value, confidence))
        
        messagebox.showinfo("Éxito", f"Extraídos {len(extractions)} campos de datos")
    
    def export_json(self):
        """Exporta los datos a JSON"""
        if not self.standardized_data:
            messagebox.showwarning("Advertencia", "No hay datos para exportar")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Guardar como JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.standardized_data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Éxito", f"Datos exportados a {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar: {str(e)}")
    
    def export_excel(self):
        """Exporta los datos a Excel"""
        if not self.standardized_data:
            messagebox.showwarning("Advertencia", "No hay datos para exportar")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Guardar como Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if file_path:
            try:
                # Convertir datos a DataFrame
                df_data = []
                for key, value in self.standardized_data.items():
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            df_data.append({'Campo': f'{key}_{i+1}', 'Valor': item})
                    else:
                        df_data.append({'Campo': key, 'Valor': value})
                
                df = pd.DataFrame(df_data)
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Éxito", f"Datos exportados a {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar: {str(e)}")

def main():
    root = tk.Tk()
    app = MedicalPrescriptionOCR(root)
    root.mainloop()

if __name__ == "__main__":
    main()