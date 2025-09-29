"""
DICCIONARIO FARMACÉUTICO INTELIGENTE
Sistema de normalización de abreviaturas y términos médicos/farmacéuticos
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# ============================================================================
# DICCIONARIO BASE - ABREVIATURAS FARMACÉUTICAS
# ============================================================================

DICCIONARIO_ABREVIATURAS = {
    # EXCIPIENTES Y BASES
    'BAO': {
        'nombre_completo': 'Base Anhidra Oleosa',
        'categoria': 'excipiente',
        'descripcion': 'Vehículo sin agua para preparaciones tópicas oleosas',
        'uso_comun': 'Pomadas, ungüentos lipófilos',
        'sinonimos': ['base anhidra', 'BAO']
    },
    'BAH': {
        'nombre_completo': 'Base Anhidra Hidrófila',
        'categoria': 'excipiente',
        'descripcion': 'Vehículo sin agua pero que absorbe agua',
        'uso_comun': 'Cremas emulsionadas',
        'sinonimos': ['base hidrófila']
    },
    'BHA': {
        'nombre_completo': 'Base Hidrófila Anhidra',
        'categoria': 'excipiente',
        'descripcion': 'Base que atrae agua pero no la contiene',
        'uso_comun': 'Preparaciones dermatológicas',
        'sinonimos': []
    },
    'CSP': {
        'nombre_completo': 'Cantidad Suficiente Para',
        'categoria': 'indicacion_cantidad',
        'descripcion': 'Completar hasta el volumen/peso final indicado',
        'uso_comun': 'Fórmulas líquidas y sólidas',
        'sinonimos': ['c.s.p.', 'csp', 'c.s.']
    },
    'QS': {
        'nombre_completo': 'Quantum Satis (Cantidad Suficiente)',
        'categoria': 'indicacion_cantidad',
        'descripcion': 'Cantidad necesaria según criterio',
        'uso_comun': 'Ajuste de pH, viscosidad',
        'sinonimos': ['q.s.']
    },
    
    # VÍAS DE ADMINISTRACIÓN
    'VO': {
        'nombre_completo': 'Vía Oral',
        'categoria': 'via_administracion',
        'descripcion': 'Administración por boca',
        'uso_comun': 'Cápsulas, soluciones, comprimidos',
        'sinonimos': ['v.o.', 'p.o.', 'per os']
    },
    'VT': {
        'nombre_completo': 'Vía Tópica',
        'categoria': 'via_administracion',
        'descripcion': 'Aplicación sobre la piel',
        'uso_comun': 'Cremas, geles, pomadas',
        'sinonimos': ['v.t.', 'tópico']
    },
    'SC': {
        'nombre_completo': 'Vía Subcutánea',
        'categoria': 'via_administracion',
        'descripcion': 'Inyección bajo la piel',
        'uso_comun': 'Inyectables',
        'sinonimos': ['s.c.', 'subcutánea']
    },
    'IM': {
        'nombre_completo': 'Vía Intramuscular',
        'categoria': 'via_administracion',
        'descripcion': 'Inyección en músculo',
        'uso_comun': 'Inyectables de depósito',
        'sinonimos': ['i.m.']
    },
    'IV': {
        'nombre_completo': 'Vía Intravenosa',
        'categoria': 'via_administracion',
        'descripcion': 'Inyección directa en vena',
        'uso_comun': 'Medicación hospitalaria',
        'sinonimos': ['i.v.', 'endovenosa']
    },
    
    # POSOLOGÍA
    'QD': {
        'nombre_completo': 'Quaque Die (Una vez al día)',
        'categoria': 'frecuencia',
        'descripcion': 'Tomar una vez cada 24 horas',
        'uso_comun': 'Posología diaria',
        'sinonimos': ['q.d.', '1/día', 'sid']
    },
    'BID': {
        'nombre_completo': 'Bis In Die (Dos veces al día)',
        'categoria': 'frecuencia',
        'descripcion': 'Tomar cada 12 horas',
        'uso_comun': 'Posología cada 12h',
        'sinonimos': ['b.i.d.', '2/día', 'c/12h']
    },
    'TID': {
        'nombre_completo': 'Ter In Die (Tres veces al día)',
        'categoria': 'frecuencia',
        'descripcion': 'Tomar cada 8 horas',
        'uso_comun': 'Posología cada 8h',
        'sinonimos': ['t.i.d.', '3/día', 'c/8h']
    },
    'QID': {
        'nombre_completo': 'Quater In Die (Cuatro veces al día)',
        'categoria': 'frecuencia',
        'descripcion': 'Tomar cada 6 horas',
        'uso_comun': 'Posología cada 6h',
        'sinonimos': ['q.i.d.', '4/día', 'c/6h']
    },
    
    # FORMAS FARMACÉUTICAS
    'caps': {
        'nombre_completo': 'Cápsulas',
        'categoria': 'forma_farmaceutica',
        'descripcion': 'Forma sólida de gelatina con polvo/líquido',
        'uso_comun': 'Administración oral',
        'sinonimos': ['cápsula', 'caps.']
    },
    'comp': {
        'nombre_completo': 'Comprimidos',
        'categoria': 'forma_farmaceutica',
        'descripcion': 'Forma sólida comprimida',
        'uso_comun': 'Administración oral',
        'sinonimos': ['comprimido', 'comp.', 'tableta']
    },
    'sol': {
        'nombre_completo': 'Solución',
        'categoria': 'forma_farmaceutica',
        'descripcion': 'Líquido con principio activo disuelto',
        'uso_comun': 'Oral, tópica, parenteral',
        'sinonimos': ['sol.', 'solución']
    },
    'susp': {
        'nombre_completo': 'Suspensión',
        'categoria': 'forma_farmaceutica',
        'descripcion': 'Líquido con sólidos en suspensión',
        'uso_comun': 'Oral, tópica',
        'sinonimos': ['susp.', 'suspensión']
    },
    
    # UNIDADES DE MEDIDA
    'mg': {
        'nombre_completo': 'Miligramo',
        'categoria': 'unidad_peso',
        'descripcion': '0.001 gramos',
        'uso_comun': 'Peso de principio activo',
        'sinonimos': ['miligramo', 'miligramos']
    },
    'mcg': {
        'nombre_completo': 'Microgramo',
        'categoria': 'unidad_peso',
        'descripcion': '0.001 miligramos',
        'uso_comun': 'Dosis muy pequeñas',
        'sinonimos': ['µg', 'microgramo']
    },
    'UI': {
        'nombre_completo': 'Unidades Internacionales',
        'categoria': 'unidad_biologica',
        'descripcion': 'Medida de actividad biológica',
        'uso_comun': 'Vitaminas, hormonas',
        'sinonimos': ['u.i.', 'IU']
    },
    
    # INSTRUCCIONES ESPECÍFICAS
    'aa': {
        'nombre_completo': 'Ana (partes iguales)',
        'categoria': 'instruccion',
        'descripcion': 'Tomar cantidades iguales de cada componente',
        'uso_comun': 'Fórmulas con múltiples principios',
        'sinonimos': ['āā', 'partes iguales']
    },
    'ad': {
        'nombre_completo': 'Ad (hasta)',
        'categoria': 'instruccion',
        'descripcion': 'Completar hasta la cantidad indicada',
        'uso_comun': 'Volumen final',
        'sinonimos': ['hasta']
    },
    'div': {
        'nombre_completo': 'Dividir',
        'categoria': 'instruccion',
        'descripcion': 'Dividir en partes iguales',
        'uso_comun': 'Preparación de dosis individuales',
        'sinonimos': ['dividir en', 'div.']
    },
    
    # TÉRMINOS ESPECÍFICOS DERMATOLOGÍA/TRICOLOGÍA
    'lot': {
        'nombre_completo': 'Loción',
        'categoria': 'forma_farmaceutica',
        'descripcion': 'Preparación líquida para aplicación cutánea',
        'uso_comun': 'Uso dermatológico/capilar',
        'sinonimos': ['loción', 'lot.']
    },
    'ung': {
        'nombre_completo': 'Ungüento',
        'categoria': 'forma_farmaceutica',
        'descripcion': 'Preparación semisólida grasa',
        'uso_comun': 'Uso dermatológico',
        'sinonimos': ['ungüento', 'ung.', 'pomada']
    },
    
    # CONSERVANTES Y ADITIVOS
    'NipSod': {
        'nombre_completo': 'Nipasol Sódico (Metilparabeno sódico)',
        'categoria': 'conservante',
        'descripcion': 'Conservante antimicrobiano',
        'uso_comun': 'Preparaciones acuosas',
        'sinonimos': ['metilparabeno', 'nipagin']
    },
    'BHT': {
        'nombre_completo': 'Butilhidroxitolueno',
        'categoria': 'antioxidante',
        'descripcion': 'Antioxidante liposoluble',
        'uso_comun': 'Prevención de oxidación',
        'sinonimos': ['butilhidroxitolueno']
    },
    'EDTA': {
        'nombre_completo': 'Ácido Etilendiaminotetraacético',
        'categoria': 'quelante',
        'descripcion': 'Quelante de metales',
        'uso_comun': 'Estabilizador de formulaciones',
        'sinonimos': ['edta', 'edetato']
    }
}

# ============================================================================
# PRINCIPIOS ACTIVOS COMUNES EN TRICOLOGÍA
# ============================================================================

PRINCIPIOS_ACTIVOS_TRICOLOGIA = {
    'minoxidil': {
        'nombre_generico': 'Minoxidil',
        'categoria': 'vasodilatador',
        'indicaciones': ['Alopecia androgenética', 'Alopecia areata'],
        'dosis_habituales': ['2%', '5%', '10%', '15%'],
        'forma_farmaceutica': ['Solución tópica', 'Espuma', 'Loción'],
        'incompatibilidades': ['Retinoicos tópicos'],
        'variantes_escritura': ['minoxidilo', 'minoxidyl', 'minoxidl']
    },
    'finasteride': {
        'nombre_generico': 'Finasteride',
        'categoria': 'inhibidor_5alfa_reductasa',
        'indicaciones': ['Alopecia androgenética masculina', 'Hiperplasia prostática'],
        'dosis_habituales': ['0.25mg', '0.5mg', '1mg', '5mg'],
        'forma_farmaceutica': ['Cápsulas', 'Comprimidos', 'Solución tópica'],
        'incompatibilidades': [],
        'variantes_escritura': ['finasterida', 'finasterid']
    },
    'dutasteride': {
        'nombre_generico': 'Dutasteride',
        'categoria': 'inhibidor_5alfa_reductasa',
        'indicaciones': ['Alopecia androgenética', 'Hiperplasia prostática'],
        'dosis_habituales': ['0.1mg', '0.25mg', '0.5mg'],
        'forma_farmaceutica': ['Cápsulas', 'Solución tópica'],
        'incompatibilidades': [],
        'variantes_escritura': ['dutasterida', 'dutasterid']
    },
    'latanoprost': {
        'nombre_generico': 'Latanoprost',
        'categoria': 'prostaglandina',
        'indicaciones': ['Crecimiento de pestañas', 'Alopecia areata cejas'],
        'dosis_habituales': ['0.005%', '0.01%'],
        'forma_farmaceutica': ['Solución oftálmica adaptada'],
        'incompatibilidades': [],
        'variantes_escritura': ['latanoprosto']
    },
    'biotina': {
        'nombre_generico': 'Biotina (Vitamina B7)',
        'categoria': 'vitamina',
        'indicaciones': ['Fortalecimiento capilar', 'Uñas quebradizas'],
        'dosis_habituales': ['2.5mg', '5mg', '10mg'],
        'forma_farmaceutica': ['Cápsulas', 'Comprimidos'],
        'incompatibilidades': [],
        'variantes_escritura': ['biotin', 'vitamina h']
    }
}

# ============================================================================
# CLASE GESTOR DEL DICCIONARIO
# ============================================================================

class PharmaDictionary:
    """
    Sistema inteligente de diccionario farmacéutico
    Expande abreviaturas, normaliza términos, detecta errores
    """
    
    def __init__(self, db_path="pharma_dictionary.db"):
        self.db_path = db_path
        self.init_database()
        self.cargar_diccionario_base()
    
    def init_database(self):
        """Inicializa la base de datos del diccionario"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de abreviaturas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS abreviaturas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                abreviatura TEXT UNIQUE NOT NULL,
                nombre_completo TEXT NOT NULL,
                categoria TEXT,
                descripcion TEXT,
                uso_comun TEXT,
                frecuencia_uso INTEGER DEFAULT 0,
                ultima_vez TIMESTAMP
            )
        """)
        
        # Tabla de sinónimos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sinonimos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                termino_principal TEXT,
                sinonimo TEXT,
                FOREIGN KEY (termino_principal) REFERENCES abreviaturas(abreviatura)
            )
        """)
        
        # Tabla de principios activos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS principios_activos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre_generico TEXT UNIQUE NOT NULL,
                categoria TEXT,
                indicaciones TEXT,
                dosis_habituales TEXT,
                formas_farmaceuticas TEXT,
                incompatibilidades TEXT,
                frecuencia_uso INTEGER DEFAULT 0
            )
        """)
        
        # Tabla de variantes ortográficas (errores comunes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variantes_ortograficas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                termino_incorrecto TEXT,
                termino_correcto TEXT,
                veces_corregido INTEGER DEFAULT 1,
                UNIQUE(termino_incorrecto, termino_correcto)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def cargar_diccionario_base(self):
        """Carga el diccionario base si está vacío"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verificar si ya hay datos
        cursor.execute("SELECT COUNT(*) FROM abreviaturas")
        if cursor.fetchone()[0] == 0:
            # Cargar abreviaturas
            for abrev, datos in DICCIONARIO_ABREVIATURAS.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO abreviaturas 
                    (abreviatura, nombre_completo, categoria, descripcion, uso_comun)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    abrev,
                    datos['nombre_completo'],
                    datos['categoria'],
                    datos['descripcion'],
                    datos['uso_comun']
                ))
                
                # Cargar sinónimos
                for sinonimo in datos.get('sinonimos', []):
                    cursor.execute("""
                        INSERT OR IGNORE INTO sinonimos (termino_principal, sinonimo)
                        VALUES (?, ?)
                    """, (abrev, sinonimo))
        
        # Verificar principios activos
        cursor.execute("SELECT COUNT(*) FROM principios_activos")
        if cursor.fetchone()[0] == 0:
            for nombre, datos in PRINCIPIOS_ACTIVOS_TRICOLOGIA.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO principios_activos
                    (nombre_generico, categoria, indicaciones, dosis_habituales, 
                     formas_farmaceuticas, incompatibilidades)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datos['nombre_generico'],
                    datos['categoria'],
                    json.dumps(datos['indicaciones']),
                    json.dumps(datos['dosis_habituales']),
                    json.dumps(datos['forma_farmaceutica']),
                    json.dumps(datos['incompatibilidades'])
                ))
                
                # Cargar variantes ortográficas
                for variante in datos.get('variantes_escritura', []):
                    cursor.execute("""
                        INSERT OR IGNORE INTO variantes_ortograficas
                        (termino_incorrecto, termino_correcto)
                        VALUES (?, ?)
                    """, (variante, datos['nombre_generico']))
        
        conn.commit()
        conn.close()
    
    def expandir_abreviaturas(self, texto: str) -> Tuple[str, List[Dict]]:
        """
        Detecta y expande abreviaturas en el texto
        Retorna: (texto_expandido, lista_abreviaturas_encontradas)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        texto_expandido = texto
        abreviaturas_encontradas = []
        
        # Buscar todas las abreviaturas en el texto
        cursor.execute("SELECT abreviatura, nombre_completo, categoria, descripcion FROM abreviaturas")
        
        for abrev, nombre_completo, categoria, descripcion in cursor.fetchall():
            # Buscar abreviatura (case insensitive, con límites de palabra)
            pattern = r'\b' + re.escape(abrev) + r'\b'
            matches = list(re.finditer(pattern, texto, re.IGNORECASE))
            
            if matches:
                # Reemplazar en texto
                texto_expandido = re.sub(
                    pattern, 
                    f"{abrev} ({nombre_completo})", 
                    texto_expandido, 
                    flags=re.IGNORECASE
                )
                
                # Guardar info
                abreviaturas_encontradas.append({
                    'abreviatura': abrev,
                    'nombre_completo': nombre_completo,
                    'categoria': categoria,
                    'descripcion': descripcion,
                    'posiciones': [m.start() for m in matches]
                })
                
                # Incrementar contador de uso
                cursor.execute("""
                    UPDATE abreviaturas 
                    SET frecuencia_uso = frecuencia_uso + ?, 
                        ultima_vez = CURRENT_TIMESTAMP
                    WHERE abreviatura = ?
                """, (len(matches), abrev))
        
        conn.commit()
        conn.close()
        
        return texto_expandido, abreviaturas_encontradas
    
    def corregir_ortografia(self, termino: str) -> Optional[str]:
        """Busca y corrige errores ortográficos comunes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buscar en variantes ortográficas
        cursor.execute("""
            SELECT termino_correcto 
            FROM variantes_ortograficas 
            WHERE LOWER(termino_incorrecto) = LOWER(?)
            ORDER BY veces_corregido DESC
            LIMIT 1
        """, (termino,))
        
        resultado = cursor.fetchone()
        conn.close()
        
        return resultado[0] if resultado else None
    
    def obtener_info_principio_activo(self, nombre: str) -> Optional[Dict]:
        """Obtiene información completa de un principio activo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT nombre_generico, categoria, indicaciones, dosis_habituales,
                   formas_farmaceuticas, incompatibilidades
            FROM principios_activos
            WHERE LOWER(nombre_generico) = LOWER(?)
        """, (nombre,))
        
        resultado = cursor.fetchone()
        conn.close()
        
        if resultado:
            return {
                'nombre_generico': resultado[0],
                'categoria': resultado[1],
                'indicaciones': json.loads(resultado[2]),
                'dosis_habituales': json.loads(resultado[3]),
                'formas_farmaceuticas': json.loads(resultado[4]),
                'incompatibilidades': json.loads(resultado[5])
            }
        
        return None
    
    def agregar_termino_personalizado(self, abreviatura: str, nombre_completo: str, 
                                     categoria: str, descripcion: str = ""):
        """Permite al usuario agregar términos personalizados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO abreviaturas
            (abreviatura, nombre_completo, categoria, descripcion, uso_comun)
            VALUES (?, ?, ?, ?, ?)
        """, (abreviatura, nombre_completo, categoria, descripcion, "Usuario personalizado"))
        
        conn.commit()
        conn.close()
    
    def obtener_estadisticas(self) -> Dict:
        """Estadísticas del diccionario"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM abreviaturas")
        total_abreviaturas = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM principios_activos")
        total_principios = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM variantes_ortograficas")
        total_variantes = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT abreviatura, nombre_completo, frecuencia_uso 
            FROM abreviaturas 
            ORDER BY frecuencia_uso DESC 
            LIMIT 10
        """)
        mas_usadas = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_abreviaturas': total_abreviaturas,
            'total_principios_activos': total_principios,
            'total_variantes': total_variantes,
            'mas_usadas': [
                {'abrev': row[0], 'nombre': row[1], 'usos': row[2]} 
                for row in mas_usadas
            ]
        }

# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Inicializar diccionario
    diccionario = PharmaDictionary()
    
    # Texto de ejemplo de receta
    texto_receta = """
    Minoxidil 5mg
    Finasteride 1mg
    BAO csp 100 caps
    
    Posología: 1 caps BID VO
    Duración: 3 meses
    """
    
    print("=== TEXTO ORIGINAL ===")
    print(texto_receta)
    
    # Expandir abreviaturas
    texto_expandido, abrevs = diccionario.expandir_abreviaturas(texto_receta)
    
    print("\n=== TEXTO EXPANDIDO ===")
    print(texto_expandido)
    
    print("\n=== ABREVIATURAS ENCONTRADAS ===")
    for abrev in abrevs:
        print(f"- {abrev['abreviatura']}: {abrev['nombre_completo']}")
        print(f"  Categoría: {abrev['categoria']}")
        print(f"  {abrev['descripcion']}\n")
    
    # Obtener info de principio activo
    print("=== INFO MINOXIDIL ===")
    info = diccionario.obtener_info_principio_activo("minoxidil")
    if info:
        print(f"Categoría: {info['categoria']}")
        print(f"Indicaciones: {', '.join(info['indicaciones'])}")
        print(f"Dosis habituales: {', '.join(info['dosis_habituales'])}")
    
    # Estadísticas
    print("\n=== ESTADÍSTICAS ===")
    stats = diccionario.obtener_estadisticas()
    print(f"Total abreviaturas: {stats['total_abreviaturas']}")
    print(f"Total principios activos: {stats['total_principios_activos']}")