# 🧪 Scripts de Prueba

Esta carpeta contiene scripts para probar el sistema de datos.

## 📁 Archivos Disponibles

### **test_data_manager.py**
Prueba básica del DataManager:
- Inicialización
- Descarga de datos
- Cálculo de retornos
- Creación de universos

**Uso:**
```bash
cd projects/data/test
python test_data_manager.py
```

### **test_universes.py**
Prueba los universos predefinidos:
- equity_us.txt
- conservative_portfolio.txt
- moderate_portfolio.txt
- aggressive_portfolio.txt
- benchmarks.txt

**Uso:**
```bash
cd projects/data/test
python test_universes.py
```

### **test_flexibility.py**
Demuestra la flexibilidad del sistema:
- Creación de diferentes tipos de universos
- Prueba de diferentes clases de activos
- Listado de todos los universos disponibles

**Uso:**
```bash
cd projects/data/test
python test_flexibility.py
```

## 🚀 Ejecutar Todas las Pruebas

```bash
cd projects/data/test

# Prueba básica
python test_data_manager.py

# Prueba de universos
python test_universes.py

# Prueba de flexibilidad
python test_flexibility.py
```

## 📊 Qué Probar

1. **Funcionalidad básica**: Descarga de datos, cálculo de retornos
2. **Universos predefinidos**: Carga y uso de archivos de universo
3. **Flexibilidad**: Creación de nuevos universos y tipos de activos
4. **Robustez**: Manejo de errores y diferentes tipos de datos

## ⚠️ Notas

- Los scripts usan rutas relativas desde `projects/data/`
- Se recomienda ejecutar desde la carpeta `test/`
- Los datos se almacenan en caché para evitar descargas repetidas
