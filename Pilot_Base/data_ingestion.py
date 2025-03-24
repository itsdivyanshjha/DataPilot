import pandas as pd
import json
import sqlite3
import yaml
import xml.etree.ElementTree as ET
from typing import Union, Dict, Any, List
import streamlit as st

class DataIngestionManager:
    SUPPORTED_ENCODINGS = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    POTENTIAL_DELIMITERS = [',', ';', '\t', '|']
    
    @staticmethod
    def read_csv(file) -> pd.DataFrame:
        """
        Read CSV file with automatic encoding detection and delimiter inference.
        Handles various CSV formats and provides detailed error messages.
        """
        try:
            df = pd.read_csv(file)
            if len(df.columns) > 0:
                return df
        except UnicodeDecodeError:
            for encoding in DataIngestionManager.SUPPORTED_ENCODINGS:
                try:
                    df = pd.read_csv(file, encoding=encoding)
                    if len(df.columns) > 0:
                        return df
                except:
                    continue
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file appears to be empty.")
        except Exception as e:
            try:
                file.seek(0)
                content = file.read().decode('utf-8')
                first_lines = content.split('\n')[:5]
                
                for delimiter in DataIngestionManager.POTENTIAL_DELIMITERS:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, sep=delimiter)
                        if len(df.columns) > 0:
                            return df
                    except:
                        continue
                
                raise ValueError(
                    f"Could not determine the correct delimiter. File preview:\n{chr(10).join(first_lines)}"
                )
            except Exception as inner_e:
                raise ValueError(
                    f"Error reading CSV: {str(e)}\nDetails: {str(inner_e)}\n"
                    "Check file validity, headers, and content."
                )
        
        raise ValueError("Could not read the CSV file. File might be empty or corrupted.")

    @staticmethod
    def read_json(file) -> pd.DataFrame:
        """Read JSON file with support for different structures"""
        try:
            data = json.load(file)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.json_normalize(data)
        except:
            return pd.read_json(file, lines=True)

    @staticmethod
    def read_excel(file) -> pd.DataFrame:
        """Read Excel file with support for multiple sheets"""
        xls = pd.ExcelFile(file)
        if len(xls.sheet_names) == 1:
            return pd.read_excel(file)
        sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
        return pd.read_excel(file, sheet_name=sheet_name)

    @staticmethod
    def read_xml(file) -> pd.DataFrame:
        """Read XML file and convert to DataFrame"""
        tree = ET.parse(file)
        root = tree.getroot()
        data = [{elem.tag: elem.text for elem in child} for child in root]
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