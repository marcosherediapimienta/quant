import ast
from pathlib import Path
from typing import List, Dict, Optional

try:
    from .tools.config import (
        DEFAULT_EXCLUDE_DIRS, DEFAULT_EXCLUDE_FILES,
        MAX_CLASS_LINES, MAX_FUNCTION_LINES, MAX_METHOD_LINES,
        MIN_METHOD_NAME_LENGTH, KEYWORD_SCORES,
    )
except ImportError:
    from tools.config import (
        DEFAULT_EXCLUDE_DIRS, DEFAULT_EXCLUDE_FILES,
        MAX_CLASS_LINES, MAX_FUNCTION_LINES, MAX_METHOD_LINES,
        MIN_METHOD_NAME_LENGTH, KEYWORD_SCORES,
    )

class CodeIndexer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.documents: List[Dict] = []

    def index_project(
        self,
        extensions: List[str] = None,
        exclude_dirs: List[str] = None,
        exclude_files: List[str] = None
    ) -> List[Dict]:

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
                # Check if it's in an excluded directory
                path_parts = set(file_path.parts)
                if path_parts & excluded_dirs:
                    continue

                # Check if the file should be excluded
                if any(excl in file_path.name for excl in excluded_files):
                    continue

                try:
                    self._index_file(file_path)
                except Exception as e:
                    print(f"[!] Error indexing {file_path}: {e}")

        print(f"[OK] Indexed {len(self.documents)} code segments")
        return self.documents

    def _index_file(self, file_path: Path):
        content = None
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"[!] Could not read {file_path} (unknown encoding)")
            return

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"[!] Syntax error in {file_path}: {e}")
            return

        lines = content.splitlines()
        rel_path = str(file_path.relative_to(self.project_root))
        module_doc = ast.get_docstring(tree)

        if module_doc:
            self.documents.append({
                'content': module_doc,
                'type': 'module',
                'name': file_path.stem,
                'file': rel_path,
                'metadata': {'docstring': module_doc, 'line_start': 1}
            })

        for node in ast.iter_child_nodes(tree):

            if isinstance(node, ast.ClassDef):
                self._index_class(node, lines, rel_path)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_function(node, lines, rel_path)

    def _index_class(self, node: ast.ClassDef, lines: List[str], rel_path: str):
        docstring = ast.get_docstring(node) or ''
        start = node.lineno - 1
        end = min(node.end_lineno, start + MAX_CLASS_LINES)
        body = '\n'.join(lines[start:end])
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

        for child in ast.iter_child_nodes(node):

            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):

                if not child.name.startswith('_') or child.name in ('__init__', '__call__', '__str__'):
                    self._index_method(child, lines, rel_path, class_name=node.name)

    def _index_function(
        self,
        node,
        lines: List[str],
        rel_path: str,
        class_name: Optional[str] = None
    ):

        docstring = ast.get_docstring(node) or ''
        start = node.lineno - 1
        end = min(node.end_lineno, start + MAX_FUNCTION_LINES)
        body = '\n'.join(lines[start:end])
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

        docstring = ast.get_docstring(node) or ''

        if not docstring and len(node.name) < MIN_METHOD_NAME_LENGTH:
            return

        start = node.lineno - 1
        end = min(node.end_lineno, start + MAX_METHOD_LINES)
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
        texts = []
        for doc in self.documents:
            text = f"# {doc['type'].upper()}: {doc['name']}\n"
            text += f"# File: {doc['file']}\n\n"
            text += doc['content']

            if doc['metadata'].get('docstring'):
                text += f"\n\n# Documentation:\n{doc['metadata']['docstring']}"

            texts.append(text)

        return texts

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:

        keyword_lower = keyword.lower()
        results = []

        for doc in self.documents:
            fields = {
                'name': doc['name'].lower(),
                'docstring': doc['metadata'].get('docstring', '').lower(),
                'content': doc['content'].lower(),
            }
            score = sum(
                weight for field, weight in KEYWORD_SCORES.items()
                if keyword_lower in fields.get(field, '')
            )

            if score > 0:
                results.append({'doc': doc, 'score': score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['doc'] for r in results[:limit]]

    def get_stats(self) -> Dict:
        stats = {'total': len(self.documents), 'by_type': {}}
        for doc in self.documents:
            doc_type = doc['type']
            stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
        return stats
