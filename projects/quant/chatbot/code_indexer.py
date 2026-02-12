"""
Indexa el código fuente del proyecto para RAG (Retrieval Augmented Generation)
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import re


class CodeIndexer:
    """
    Indexa archivos de código Python para búsqueda semántica
    """
    
    def __init__(self, project_root: str):
        """
        Args:
            project_root: Ruta raíz del proyecto a indexar
        """
        self.project_root = Path(project_root)
        self.documents: List[Dict] = []
    
    def index_project(self, extensions: List[str] = ['.py'], exclude_dirs: List[str] = None):
        """
        Indexa archivos del proyecto
        
        Args:
            extensions: Extensiones de archivo a indexar
            exclude_dirs: Directorios a excluir
        """
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__', '.git', 'venv', 'env', '.pytest_cache']
        
        self.documents = []
        
        for ext in extensions:
            for file_path in self.project_root.rglob(f'*{ext}'):
                # Verificar si está en directorio excluido
                if any(excluded in str(file_path) for excluded in exclude_dirs):
                    continue
                
                try:
                    self._index_file(file_path)
                except Exception as e:
                    print(f"Error indexando {file_path}: {e}")
        
        print(f"✓ Indexados {len(self.documents)} segmentos de código")
        return self.documents
    
    def _index_file(self, file_path: Path):
        """Indexa un archivo individual"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extraer clases y funciones
        classes = self._extract_classes(content)
        functions = self._extract_functions(content)
        
        # Ruta relativa para metadata
        rel_path = file_path.relative_to(self.project_root)
        
        # Indexar clases
        for cls in classes:
            self.documents.append({
                'content': cls['content'],
                'type': 'class',
                'name': cls['name'],
                'file': str(rel_path),
                'metadata': {
                    'docstring': cls.get('docstring', ''),
                    'line_start': cls.get('line_start', 0)
                }
            })
        
        # Indexar funciones
        for func in functions:
            self.documents.append({
                'content': func['content'],
                'type': 'function',
                'name': func['name'],
                'file': str(rel_path),
                'metadata': {
                    'docstring': func.get('docstring', ''),
                    'line_start': func.get('line_start', 0)
                }
            })
        
        # Si el archivo tiene docstring al inicio, indexarlo también
        module_doc = self._extract_module_docstring(content)
        if module_doc:
            self.documents.append({
                'content': module_doc,
                'type': 'module',
                'name': rel_path.stem,
                'file': str(rel_path),
                'metadata': {}
            })
    
    def _extract_classes(self, content: str) -> List[Dict]:
        """Extrae definiciones de clases"""
        classes = []
        pattern = r'class\s+(\w+).*?:\s*(?:"""(.*?)""")?(.*?)(?=\nclass\s|\ndef\s|\Z)'
        
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            class_name = match.group(1)
            docstring = match.group(2) or ''
            body = match.group(3) or ''
            
            # Reconstruir la definición completa
            full_content = f"class {class_name}:\n"
            if docstring:
                full_content += f'    """{docstring}"""\n'
            full_content += body[:500]  # Limitar tamaño
            
            classes.append({
                'name': class_name,
                'content': full_content,
                'docstring': docstring.strip(),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def _extract_functions(self, content: str) -> List[Dict]:
        """Extrae definiciones de funciones"""
        functions = []
        pattern = r'def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*[\w\[\],\s]+)?:\s*(?:"""(.*?)""")?(.*?)(?=\ndef\s|\nclass\s|\Z)'
        
        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            func_name = match.group(1)
            params = match.group(2)
            docstring = match.group(3) or ''
            body = match.group(4) or ''
            
            # Reconstruir la definición
            full_content = f"def {func_name}({params}):\n"
            if docstring:
                full_content += f'    """{docstring}"""\n'
            full_content += body[:300]  # Limitar tamaño
            
            functions.append({
                'name': func_name,
                'content': full_content,
                'docstring': docstring.strip(),
                'line_start': content[:match.start()].count('\n') + 1
            })
        
        return functions
    
    def _extract_module_docstring(self, content: str) -> Optional[str]:
        """Extrae docstring del módulo (al inicio del archivo)"""
        pattern = r'^"""(.*?)"""'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def get_documents_text(self) -> List[str]:
        """Obtiene lista de textos para embeddings"""
        texts = []
        for doc in self.documents:
            # Combinar nombre, tipo, archivo y contenido para mejor contexto
            text = f"# {doc['type'].upper()}: {doc['name']}\n"
            text += f"# Archivo: {doc['file']}\n\n"
            text += doc['content']
            
            if doc['metadata'].get('docstring'):
                text += f"\n\n# Documentación:\n{doc['metadata']['docstring']}"
            
            texts.append(text)
        
        return texts
    
    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:
        """
        Búsqueda simple por palabra clave (fallback si no hay embeddings)
        
        Args:
            keyword: Término a buscar
            limit: Número máximo de resultados
        """
        keyword_lower = keyword.lower()
        results = []
        
        for doc in self.documents:
            # Buscar en nombre, contenido y docstring
            score = 0
            if keyword_lower in doc['name'].lower():
                score += 10
            if keyword_lower in doc['content'].lower():
                score += 5
            if keyword_lower in doc['metadata'].get('docstring', '').lower():
                score += 3
            
            if score > 0:
                results.append({'doc': doc, 'score': score})
        
        # Ordenar por score y devolver top resultados
        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['doc'] for r in results[:limit]]
