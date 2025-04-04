import pandas as pd
import json
import sqlite3
import yaml
import xml.etree.ElementTree as ET
from typing import Union, Dict, Any, List
import streamlit as st
import io

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
            # Convert Streamlit UploadedFile to BytesIO if needed
            if hasattr(file, 'read'):
                file_content = file.read()
                file_buffer = io.BytesIO(file_content)
            else:
                file_buffer = file

            # Try reading with default settings first
            try:
                df = pd.read_csv(file_buffer)
                if len(df.columns) > 0:
                    return df
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in DataIngestionManager.SUPPORTED_ENCODINGS:
                    try:
                        file_buffer.seek(0)  # Reset buffer position
                        df = pd.read_csv(file_buffer, encoding=encoding)
                        if len(df.columns) > 0:
                            return df
                    except:
                        continue
                raise ValueError(f"Could not read file with any of the supported encodings: {DataIngestionManager.SUPPORTED_ENCODINGS}")
            
            # If we get here, try different delimiters
            file_buffer.seek(0)
            content = file_buffer.read().decode('utf-8', errors='ignore')
            first_lines = content.split('\n')[:5]
            
            for delimiter in DataIngestionManager.POTENTIAL_DELIMITERS:
                try:
                    file_buffer.seek(0)
                    df = pd.read_csv(file_buffer, sep=delimiter)
                    if len(df.columns) > 0:
                        return df
                except:
                    continue
            
            raise ValueError(
                f"Could not determine the correct delimiter. File preview:\n{chr(10).join(first_lines)}"
            )
            
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file appears to be empty.")
        except Exception as e:
            raise ValueError(f"Error reading CSV: {str(e)}\nPlease check file validity, headers, and content.")

    @staticmethod
    def read_json(file) -> pd.DataFrame:
        """Read JSON file with support for different structures"""
        try:
            # Convert Streamlit UploadedFile to BytesIO if needed
            if hasattr(file, 'read'):
                file_content = file.read()
                file_buffer = io.BytesIO(file_content)
            else:
                file_buffer = file

            # Try reading as regular JSON first
            try:
                data = json.load(file_buffer)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                return pd.json_normalize(data)
            except:
                # If that fails, try reading as JSON lines
                file_buffer.seek(0)
                return pd.read_json(file_buffer, lines=True)
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")

    @staticmethod
    def read_excel(file) -> pd.DataFrame:
        """Read Excel file with support for multiple sheets"""
        try:
            # Convert Streamlit UploadedFile to BytesIO if needed
            if hasattr(file, 'read'):
                file_content = file.read()
                file_buffer = io.BytesIO(file_content)
            else:
                file_buffer = file

            xls = pd.ExcelFile(file_buffer)
            if len(xls.sheet_names) == 1:
                return pd.read_excel(file_buffer)
            sheet_name = st.selectbox("Select Sheet", xls.sheet_names)
            return pd.read_excel(file_buffer, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")

    @staticmethod
    def read_xml(file) -> pd.DataFrame:
        """Read XML file and convert to DataFrame"""
        try:
            # Convert Streamlit UploadedFile to BytesIO if needed
            if hasattr(file, 'read'):
                file_content = file.read()
                file_buffer = io.BytesIO(file_content)
            else:
                file_buffer = file

            tree = ET.parse(file_buffer)
            root = tree.getroot()
            data = [{elem.tag: elem.text for elem in child} for child in root]
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Error reading XML file: {str(e)}")

    @staticmethod
    def read_sqlite(file) -> Dict[str, pd.DataFrame]:
        """Read SQLite database and return dict of DataFrames"""
        try:
            # For SQLite, we need to save the uploaded file temporarily
            if hasattr(file, 'read'):
                file_content = file.read()
                temp_path = "temp_database.db"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                conn = sqlite3.connect(temp_path)
            else:
                conn = sqlite3.connect(file)

            # Get all table names
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            result = {table: pd.read_sql(f"SELECT * FROM {table}", conn) for table in tables['name']}
            
            # Close connection and clean up temporary file if created
            conn.close()
            if hasattr(file, 'read'):
                import os
                os.remove(temp_path)
            
            return result
        except Exception as e:
            raise ValueError(f"Error reading SQLite database: {str(e)}")

    @staticmethod
    def read_yaml(file) -> pd.DataFrame:
        """Read YAML file and convert to DataFrame"""
        try:
            # Convert Streamlit UploadedFile to BytesIO if needed
            if hasattr(file, 'read'):
                file_content = file.read()
                file_buffer = io.BytesIO(file_content)
            else:
                file_buffer = file

            data = yaml.safe_load(file_buffer)
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Error reading YAML file: {str(e)}")

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
        
        try:
            return readers[file_type](file)
        except Exception as e:
            raise ValueError(f"Error processing {file_type} file: {str(e)}")

    @staticmethod
    def get_file_info(df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the DataFrame"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'dtypes': df.dtypes.to_dict()
        } 