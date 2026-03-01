# ./src/ollamatoolkit/tools/files.py
"""
Ollama Toolkit - File Tools
===========================
Safe file system operations supporting Text, JSON, CSV, and Markdown.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union


class FileTools:
    """
    Safe file operations scoped to a root directory.
    """

    def __init__(self, root_dir: str = ".", read_only: bool = False):
        self.root_dir = Path(root_dir).resolve()
        self.read_only = read_only

        if not self.root_dir.exists():
            try:
                self.root_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass  # Might be generic path validation

    def _validate_path(self, path_str: str) -> Path:
        """Ensures path is within root_dir."""
        target = (self.root_dir / path_str).resolve()
        # Security: Prevent traversal
        if not str(target).startswith(str(self.root_dir)):
            raise ValueError(
                f"Access denied: Path '{path_str}' escapes root '{self.root_dir}'"
            )
        return target

    def _check_write(self):
        if self.read_only:
            raise PermissionError("FileTools is in READ_ONLY mode.")

    # --- Text Operations ---
    def read_text(self, path: str) -> str:
        """Reads plain text from a file."""
        try:
            target = self._validate_path(path)
            if not target.exists():
                return f"Error: File {path} not found."
            return target.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading text: {e}"

    def write_text(self, path: str, content: str, overwrite: bool = False) -> str:
        """Writes plain text to a file."""
        try:
            self._check_write()
            target = self._validate_path(path)
            if target.exists() and not overwrite:
                return f"Error: File '{path}' exists. Set overwrite=True."

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Success: Wrote to {path}"
        except Exception as e:
            return f"Error writing text: {e}"

    def append_text(self, path: str, content: str) -> str:
        """Appends text to a file."""
        try:
            self._check_write()
            target = self._validate_path(path)
            if not target.exists():
                return self.write_text(path, content)

            with target.open("a", encoding="utf-8") as f:
                f.write(content)
            return f"Success: Appended to {path}"
        except Exception as e:
            return f"Error appending: {e}"

    # --- JSON Operations ---
    def read_json(self, path: str) -> Union[Dict, List, str]:
        """Reads and parses a JSON file."""
        try:
            target = self._validate_path(path)
            if not target.exists():
                return f"Error: File {path} not found."
            with target.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in {path}"
        except Exception as e:
            return f"Error reading JSON: {e}"

    def write_json(
        self, path: str, data: Union[Dict, List], overwrite: bool = False
    ) -> str:
        """Writes data as pretty-printed JSON."""
        try:
            self._check_write()
            # If data comes from LLM as string, try to parse it first?
            # Ideally the agent passes a Dict via python function call.
            # But if it passes a string representation, we might need ast.literal_eval or json.loads.
            # For now assume the Type is handled by the Agent framework (it loads JSON args).

            target = self._validate_path(path)
            if target.exists() and not overwrite:
                return f"Error: File '{path}' exists. Set overwrite=True."

            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return f"Success: Wrote JSON to {path}"
        except Exception as e:
            return f"Error writing JSON: {e}"

    # --- CSV Operations ---
    def read_csv(self, path: str) -> Union[List[Dict[str, str]], str]:
        """Reads a CSV file into a list of dictionaries."""
        try:
            target = self._validate_path(path)
            if not target.exists():
                return f"Error: File {path} not found."

            with target.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            return f"Error reading CSV: {e}"

    def write_csv(
        self, path: str, data: List[Dict[str, Any]], overwrite: bool = False
    ) -> str:
        """Writes a list of dictionaries to a CSV file."""
        try:
            self._check_write()
            if not data:
                return "Error: No data to write."

            target = self._validate_path(path)
            if target.exists() and not overwrite:
                return f"Error: File '{path}' exists. Set overwrite=True."

            target.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = data[0].keys()
            with target.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            return f"Success: Wrote CSV to {path}"
        except Exception as e:
            return f"Error writing CSV: {e}"

    # --- Directory Operations ---
    def list_dir(self, path: str = ".") -> str:
        """Lists directory contents."""
        try:
            target = self._validate_path(path)
            if not target.is_dir():
                return f"Error: {path} is not a directory."

            items = []
            for item in target.iterdir():
                kind = "DIR" if item.is_dir() else "FILE"
                items.append(f"[{kind}] {item.name}")
            return "\n".join(items) if items else "(Empty)"
        except Exception as e:
            return f"Error listing directory: {e}"

    # --- Search Operations ---
    def find_files(
        self, pattern: str, path: str = ".", max_results: int = 100
    ) -> List[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.md", "test_*.json")
            path: Directory to search from (relative to root_dir)
            max_results: Maximum number of results to return

        Returns:
            List of matching file paths (relative to root_dir)

        Example:
            files = tools.find_files("**/*.py")  # Find all Python files
            files = tools.find_files("*.json", "config/")  # JSON in config dir
        """
        try:
            target = self._validate_path(path)
            if not target.is_dir():
                return [f"Error: {path} is not a directory."]

            matches = []
            for match in target.glob(pattern):
                if match.is_file():
                    # Return path relative to root_dir
                    rel_path = str(match.relative_to(self.root_dir))
                    matches.append(rel_path)
                    if len(matches) >= max_results:
                        break

            return matches
        except Exception as e:
            return [f"Error searching files: {e}"]

    def grep(
        self,
        pattern: str,
        path: str = ".",
        ignore_case: bool = False,
        max_results: int = 100,
        context_lines: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for text pattern in files (regex supported).

        Args:
            pattern: Regex pattern to search for
            path: File or directory to search (relative to root_dir)
            ignore_case: If True, perform case-insensitive search
            max_results: Maximum number of matches to return
            context_lines: Number of lines to include before/after match

        Returns:
            List of dicts with: file, line, content, (and context if requested)

        Example:
            # Find all TODO comments
            matches = tools.grep(r"TODO:.*", "src/")

            # Case-insensitive search with context
            matches = tools.grep("error", "logs/", ignore_case=True, context_lines=2)
        """
        import re

        try:
            target = self._validate_path(path)
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)

            results = []

            # Determine files to search
            if target.is_file():
                files_to_search = [target]
            else:
                files_to_search = list(target.rglob("*"))

            for file_path in files_to_search:
                if not file_path.is_file():
                    continue

                # Skip binary files
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                lines = content.splitlines()

                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        match_info = {
                            "file": str(file_path.relative_to(self.root_dir)),
                            "line": i,
                            "content": line.strip()[:500],  # Limit line length
                        }

                        # Add context if requested
                        if context_lines > 0:
                            start = max(0, i - 1 - context_lines)
                            end = min(len(lines), i + context_lines)
                            match_info["context"] = {
                                "before": lines[start : i - 1],
                                "after": lines[i:end],
                            }

                        results.append(match_info)

                        if len(results) >= max_results:
                            return results

            return results

        except re.error as e:
            return [{"error": f"Invalid regex pattern: {e}"}]
        except Exception as e:
            return [{"error": f"Search error: {e}"}]

    def file_stats(self, path: str) -> Dict[str, Any]:
        """
        Get detailed statistics about a file or directory.

        Args:
            path: File or directory path (relative to root_dir)

        Returns:
            Dict with size, modification time, type, and other metadata
        """
        try:
            target = self._validate_path(path)
            if not target.exists():
                return {"error": f"Path not found: {path}"}

            stat = target.stat()
            is_dir = target.is_dir()

            result = {
                "path": str(target.relative_to(self.root_dir)),
                "type": "directory" if is_dir else "file",
                "size_bytes": stat.st_size if not is_dir else None,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
            }

            if is_dir:
                # Count contents
                children = list(target.iterdir())
                result["child_count"] = len(children)
                result["file_count"] = sum(1 for c in children if c.is_file())
                result["dir_count"] = sum(1 for c in children if c.is_dir())
            else:
                # File-specific info
                result["extension"] = target.suffix
                result["size_human"] = self._format_size(stat.st_size)

                # Line count for text files
                if target.suffix in {
                    ".txt",
                    ".md",
                    ".py",
                    ".json",
                    ".csv",
                    ".yaml",
                    ".yml",
                }:
                    try:
                        result["line_count"] = len(
                            target.read_text(encoding="utf-8").splitlines()
                        )
                    except Exception:
                        pass

            return result

        except Exception as e:
            return {"error": f"Stats error: {e}"}

    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human-readable string."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
