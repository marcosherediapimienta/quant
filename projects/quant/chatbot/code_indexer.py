"""
Indexa el código fuente del proyecto para RAG (Retrieval Augmented Generation).
Usa el módulo `ast` de Python para un parsing robusto y preciso.
"""
import ast
import os
from pathlib import Path
from typing import List, Dict, Optional


# Directorios excluidos por defecto
DEFAULT_EXCLUDE_DIRS = [
    '__pycache__', '.git', 'venv', 'env', '.pytest_cache',
    'node_modules', 'dist', 'build', '.tox', 'htmlcov',
    '.mypy_cache', '.ruff_cache', 'eggs', '.eggs'
]

# Archivos excluidos por defecto
DEFAULT_EXCLUDE_FILES = [
    'setup.py', 'setup.cfg', 'conftest.py', 'manage.py',
    'wsgi.py', 'asgi.py', 'migrations'
]


class CodeIndexer:
    """
    Indexa archivos de código Python para búsqueda semántica usando AST.

    Ventajas sobre regex:
    - Maneja correctamente clases con herencia
    - No duplica funciones anidadas en clases
    - Soporta decoradores, type hints, docstrings con comillas simples
    - Robusto ante código malformado (captura SyntaxError)
    """

    def __init__(self, project_root: str):
        """
        Args:
            project_root: Ruta raíz del proyecto a indexar
        """
        self.project_root = Path(project_root)
        self.documents: List[Dict] = []

    def index_project(
        self,
        extensions: List[str] = None,
        exclude_dirs: List[str] = None,
        exclude_files: List[str] = None
    ) -> List[Dict]:
        """
        Indexa archivos del proyecto usando AST.

        Args:
            extensions: Extensiones de archivo a indexar (default: ['.py'])
            exclude_dirs: Directorios adicionales a excluir
            exclude_files: Archivos/patrones adicionales a excluir

        Returns:
            Lista de documentos indexados
        """
        if extensions is None:
            extensions = ['.py']

        excluded_dirs = set(DEFAULT_EXCLUDE_DIRS)
        if exclude_dirs:
            excluded_dirs.update(exclude_dirs)

        excluded_files = set(DEFAULT_EXCLUDE_FILES)
        if exclude_files:
            excluded_files.update(exclude_files)

        self.documents = []

        for ext in extensions:
            for file_path in self.project_root.rglob(f'*{ext}'):
                # Verificar si está en directorio excluido
                path_parts = set(file_path.parts)
                if path_parts & excluded_dirs:
                    continue

                # Verificar si el archivo debe excluirse
                if any(excl in file_path.name for excl in excluded_files):
                    continue

                try:
                    self._index_file(file_path)
                except Exception as e:
                    print(f"⚠ Error indexando {file_path}: {e}")

        print(f"✓ Indexados {len(self.documents)} segmentos de código")
        return self.documents

    def _index_file(self, file_path: Path):
        """
        Indexa un archivo individual usando AST.

        Estrategia:
        - Top-level classes: clase completa con sus métodos
        - Top-level functions: función completa
        - Module docstring: descripción del módulo
        """
        # Leer con fallback de encoding
        content = None
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"⚠ No se pudo leer {file_path} (encoding desconocido)")
            return

        # Parsear con AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"⚠ Error de sintaxis en {file_path}: {e}")
            return

        lines = content.splitlines()
        rel_path = str(file_path.relative_to(self.project_root))

        # Indexar docstring del módulo
        module_doc = ast.get_docstring(tree)
        if module_doc:
            self.documents.append({
                'content': module_doc,
                'type': 'module',
                'name': file_path.stem,
                'file': rel_path,
                'metadata': {'docstring': module_doc, 'line_start': 1}
            })

        # Iterar solo top-level nodes (no funciones anidadas dentro de clases)
        for node in ast.iter_child_nodes(tree):

            if isinstance(node, ast.ClassDef):
                self._index_class(node, lines, rel_path)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_function(node, lines, rel_path)

    def _index_class(self, node: ast.ClassDef, lines: List[str], rel_path: str):
        """
        Indexa una clase completa (incluyendo sus métodos más importantes).
        """
        docstring = ast.get_docstring(node) or ''

        # Extraer cuerpo de la clase (primeras 60 líneas para no exceder chunk size)
        start = node.lineno - 1
        end = min(node.end_lineno, start + 60)
        body = '\n'.join(lines[start:end])

        # Reconstruir signature con herencia
        bases = [ast.unparse(b) for b in node.bases] if node.bases else []
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        signature += ":"

        self.documents.append({
            'content': body,
            'type': 'class',
            'name': node.name,
            'file': rel_path,
            'metadata': {
                'docstring': docstring,
                'line_start': node.lineno,
                'signature': signature,
                'bases': bases
            }
        })

        # Indexar también los métodos públicos de la clase individualmente
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Solo métodos públicos (no _privados, excepto __init__ y __call__)
                if not child.name.startswith('_') or child.name in ('__init__', '__call__', '__str__'):
                    self._index_method(child, lines, rel_path, class_name=node.name)

    def _index_function(
        self,
        node,
        lines: List[str],
        rel_path: str,
        class_name: Optional[str] = None
    ):
        """
        Indexa una función top-level.
        """
        docstring = ast.get_docstring(node) or ''

        # Extraer cuerpo (primeras 40 líneas)
        start = node.lineno - 1
        end = min(node.end_lineno, start + 40)
        body = '\n'.join(lines[start:end])

        # Nombre completo (ej: ClassName.method_name)
        full_name = f"{class_name}.{node.name}" if class_name else node.name

        self.documents.append({
            'content': body,
            'type': 'function',
            'name': full_name,
            'file': rel_path,
            'metadata': {
                'docstring': docstring,
                'line_start': node.lineno,
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'class_name': class_name
            }
        })

    def _index_method(
        self,
        node,
        lines: List[str],
        rel_path: str,
        class_name: str
    ):
        """
        Indexa un método de clase individualmente para mejor búsqueda.
        """
        docstring = ast.get_docstring(node) or ''

        # Solo indexar métodos con docstring o con nombre significativo
        # (evitar métodos triviales sin documentación)
        if not docstring and len(node.name) < 4:
            return

        start = node.lineno - 1
        end = min(node.end_lineno, start + 30)
        body = '\n'.join(lines[start:end])

        self.documents.append({
            'content': body,
            'type': 'method',
            'name': f"{class_name}.{node.name}",
            'file': rel_path,
            'metadata': {
                'docstring': docstring,
                'line_start': node.lineno,
                'class_name': class_name,
                'is_async': isinstance(node, ast.AsyncFunctionDef)
            }
        })

    def get_documents_text(self) -> List[str]:
        """Obtiene lista de textos para embeddings"""
        texts = []
        for doc in self.documents:
            text = f"# {doc['type'].upper()}: {doc['name']}\n"
            text += f"# Archivo: {doc['file']}\n\n"
            text += doc['content']

            if doc['metadata'].get('docstring'):
                text += f"\n\n# Documentación:\n{doc['metadata']['docstring']}"

            texts.append(text)

        return texts

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:
        """
        Búsqueda simple por palabra clave (fallback si no hay embeddings).

        Args:
            keyword: Término a buscar
            limit: Número máximo de resultados
        """
        keyword_lower = keyword.lower()
        results = []

        for doc in self.documents:
            score = 0
            name_lower = doc['name'].lower()
            content_lower = doc['content'].lower()
            docstring_lower = doc['metadata'].get('docstring', '').lower()

            # Ponderación por ubicación del keyword
            if keyword_lower in name_lower:
                score += 10
            if keyword_lower in docstring_lower:
                score += 7
            if keyword_lower in content_lower:
                score += 3

            if score > 0:
                results.append({'doc': doc, 'score': score})

        # Ordenar por score y devolver top resultados
        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['doc'] for r in results[:limit]]

    def get_stats(self) -> Dict:
        """Retorna estadísticas de la indexación"""
        stats = {'total': len(self.documents), 'by_type': {}}
        for doc in self.documents:
            doc_type = doc['type']
            stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
        return stats
