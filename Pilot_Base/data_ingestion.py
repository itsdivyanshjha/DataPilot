import pandas as pd
import json
import sqlite3
import yaml
import xml.etree.ElementTree as ET
from typing import Union, Dict, Any
import streamlit as st

class DataIngestionManager:
    """Manages data ingestion from various sources and formats"""
    
    @staticmethod
    def read_csv(file) -> pd.DataFrame:
        """Read CSV file with automatic encoding detection"""
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            return pd.read_csv(file, encoding='latin1')

    @staticmethod
    def read_json(file) -> pd.DataFrame:
        """Read JSON file with support for different structures"""
        try:
            # Try reading as regular JSON
            data = json.load(file)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle nested JSON structures
                return pd.json_normalize(data)
        except Exception as e:
            # Try reading as JSON lines
            return pd.read_json(file, lines=True)

    @staticmethod
    def read_excel(file) -> pd.DataFrame:
        """Read Excel file with support for multiple sheets"""
        xls = pd.ExcelFile(file)
        if len(xls.sheet_names) == 1:
            return pd.read_excel(file)
        else:
            sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
            return pd.read_excel(file, sheet_name=sheet_name)

    @staticmethod
    def read_xml(file) -> pd.DataFrame:
        """Read XML file and convert to DataFrame"""
        tree = ET.parse(file)
        root = tree.getroot()
        data = []
        for child in root:
            data.append({elem.tag: elem.text for elem in child})
        return pd.DataFrame(data)

    @staticmethod
    def read_sqlite(file) -> Dict[str, pd.DataFrame]:
        """Read SQLite database and return dict of DataFrames"""
        conn = sqlite3.connect(file)
        # Get all table names
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        return {table: pd.read_sql(f"SELECT * FROM {table}", conn) for table in tables['name']}

    @staticmethod
    def read_yaml(file) -> pd.DataFrame:
        """Read YAML file and convert to DataFrame"""
        data = yaml.safe_load(file)
        return pd.DataFrame(data)

    @classmethod
    def read_file(cls, file, file_type: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Main method to read files based on type"""
        readers = {
            'csv': cls.read_csv,
            'json': cls.read_json,
            'xlsx': cls.read_excel,
            'xls': cls.read_excel,
            'xml': cls.read_xml,
            'db': cls.read_sqlite,
            'sqlite': cls.read_sqlite,
            'yaml': cls.read_yaml,
            'yml': cls.read_yaml
        }
        
        if file_type not in readers:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return readers[file_type](file)

    @staticmethod
    def get_file_info(df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the DataFrame"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'dtypes': df.dtypes.to_dict()
        } 