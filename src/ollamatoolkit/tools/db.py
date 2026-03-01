# ./ollamatoolkit/tools/db.py
"""
Ollama Toolkit - Database Tools
===============================
Utilities for safe SQL interaction (SQLite).
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SQLDatabaseTool:
    """
    A robust interaction layer for SQL databases.
    """

    def __init__(self, connection_string: str, read_only: bool = True):
        self.connection_string = connection_string
        self.read_only = read_only
        self._mem_conn = None  # Cache for :memory:

        # If it's a file path and not memory, ensure we can connect
        if connection_string != ":memory:":
            path = Path(connection_string)
            if not path.exists() and read_only:
                # It's okay, maybe it will fail on connect, but good to warn?
                pass

    def _get_connection(self):
        if self.connection_string == ":memory:":
            if not self._mem_conn:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._mem_conn.row_factory = sqlite3.Row
            return self._mem_conn

        # Determine strictness?
        conn = sqlite3.connect(self.connection_string)
        conn.row_factory = sqlite3.Row  # Allow dict-like access
        return conn

    def _is_safe_query(self, sql: str) -> bool:
        if not self.read_only:
            return True

        sql_upper = sql.upper().strip()
        # Basic keyword blocking for Read-Only mode
        unsafe = [
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "DROP ",
            "ALTER ",
            "TRUNCATE ",
            "REPLACE ",
            "CREATE ",
        ]
        for kw in unsafe:
            if kw in sql_upper:
                return False
        return True

    def run_query(self, sql: str) -> str:
        """Executes a single SQL query."""
        if not self._is_safe_query(sql):
            return "Error: Query blocked by READ_ONLY mode."

        try:
            # Note: For file-based DBs, we open/close. For memory, we reuse.
            # If reusing, don't use 'with' context manager if it closes the connection!
            # sqlite3 context manager commits on exit, but does NOT close connection usually.
            # However, if we return `conn` from `_get_connection`, we should handle closing if it's new.

            conn = self._get_connection()
            # We don't close memory conn. We close file conn?
            # Actually simplest is: if file, close. If memory, don't.

            cursor = conn.cursor()
            cursor.execute(sql)

            # If it's a SELECT, return data
            if sql.strip().upper().startswith(
                "SELECT"
            ) or sql.strip().upper().startswith("PRAGMA"):
                rows = cursor.fetchall()
                if self.connection_string != ":memory:":
                    conn.close()

                if not rows:
                    return "Result: [Empty]"

                # Format as list of dicts for clarity? Or formatted text?
                # List of dicts is token-heavy but unambiguous.
                results = [dict(row) for row in rows]
                return str(results)
            else:
                # It's a mutation
                if self.connection_string != ":memory:":
                    conn.commit()
                    conn.close()
                else:
                    conn.commit()  # Commit for memory too

                return "Success."

        except Exception as e:
            return f"SQL Error: {e}"

    def execute_script(self, script: str) -> str:
        """Executes a SQL script (multiple statements). Blocked in Read-Only."""
        if self.read_only:
            return "Error: SQL scripts blocked in READ_ONLY mode."

        try:
            conn = self._get_connection()
            conn.executescript(script)

            if self.connection_string != ":memory:":
                conn.close()

            return "Script executed successfully."
        except Exception as e:
            return f"Script Error: {e}"

    def get_table_info(self, table_name: str) -> str:
        """
        Returns schema AND sample rows for a table.
        Best for 'understanding' a table structure.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 1. Get Schema
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            res = cursor.fetchone()
            if not res:
                if self.connection_string != ":memory:":
                    conn.close()
                return f"Error: Table '{table_name}' not found."
            create_stmt = res["sql"]

            # 2. Get Sample (3 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = [dict(r) for r in cursor.fetchall()]

            if self.connection_string != ":memory:":
                conn.close()

            return (
                f"--- Schema ---\n{create_stmt}\n\n--- Sample Rows (Max 3) ---\n{rows}"
            )
        except Exception as e:
            return f"Error getting table info: {e}"

    def list_tables(self) -> str:
        """Lists all tables in the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            rows = cursor.fetchall()
            if self.connection_string != ":memory:":
                conn.close()
            return str([r["name"] for r in rows])
        except Exception as e:
            return f"Error listing tables: {e}"
