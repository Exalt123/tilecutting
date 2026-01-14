"""
Google Sheets Service for Tile Cutting Automation
==================================================
Handles all read/write operations with Google Sheets.
Uses Streamlit secrets for credentials - configured in Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread


class GoogleSheetsService:
    """Service class for Google Sheets operations."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self):
        """Initialize the Google Sheets service with credentials from secrets."""
        self.client = None
        self.spreadsheet_id = None
        self.sheet_name = None
        self._initialize()
    
    def _initialize(self):
        """Load credentials and configuration from Streamlit secrets."""
        try:
            # Get credentials from secrets
            if "gcp_service_account" not in st.secrets:
                return
            
            credentials_dict = dict(st.secrets["gcp_service_account"])
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=self.SCOPES
            )
            
            # Initialize gspread client
            self.client = gspread.authorize(credentials)
            
            # Get spreadsheet configuration
            self.spreadsheet_id = st.secrets.get("spreadsheet_id", "")
            self.sheet_name = st.secrets.get("sheet_name", "Tile")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize Google Sheets: {e}")
    
    def is_connected(self) -> bool:
        """Check if the service is properly connected."""
        return self.client is not None and bool(self.spreadsheet_id)
    
    @st.cache_data(ttl=60)
    def fetch_pending_orders(_self) -> pd.DataFrame:
        """
        Fetch all pending orders from Google Sheets.
        
        Only returns rows where date_scheduled is empty (not yet scheduled).
        
        Expected columns:
        - customer
        - job_number
        - tile_width
        - tile_length
        - tile_cut_width
        - tile_cut_length
        - qty_required
        - date_available
        - date_scheduled
        
        Returns:
            pd.DataFrame: DataFrame of pending orders
        """
        if not _self.is_connected():
            st.error("❌ Not connected to Google Sheets. Check your secrets configuration.")
            return pd.DataFrame()
        
        try:
            # Open spreadsheet and worksheet
            spreadsheet = _self.client.open_by_key(_self.spreadsheet_id)
            worksheet = spreadsheet.worksheet(_self.sheet_name)
            
            # Get all records as list of dicts
            records = worksheet.get_all_records()
            
            if not records:
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            
            # Normalize column names (strip whitespace)
            df.columns = df.columns.str.strip()
            
            # Filter out scheduled orders - only keep rows where date_scheduled is empty
            if 'date_scheduled' in df.columns:
                df = df[
                    (df['date_scheduled'].isna()) | 
                    (df['date_scheduled'].astype(str).str.strip() == '')
                ]
            
            # Convert numeric columns
            numeric_cols = ['tile_width', 'tile_length', 'tile_cut_width', 'tile_cut_length', 'qty_required']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Add row index for write-back (1-indexed, +1 for header)
            df['_row_index'] = range(2, len(df) + 2)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error fetching orders: {e}")
            return pd.DataFrame()
    
    def update_status_scheduled(self, row_indices: list) -> bool:
        """
        Update the date_scheduled column to current datetime for specified rows.
        
        Args:
            row_indices: List of row indices (1-indexed) to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            st.error("❌ Not connected to Google Sheets")
            return False
        
        try:
            spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            worksheet = spreadsheet.worksheet(self.sheet_name)
            
            # Get header row to find date_scheduled column
            headers = worksheet.row_values(1)
            
            if 'date_scheduled' not in headers:
                st.error("❌ 'date_scheduled' column not found in sheet")
                return False
            
            status_col = headers.index('date_scheduled') + 1  # 1-indexed
            
            # Create timestamp
            scheduled_value = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Batch update for efficiency
            cells_to_update = []
            for row_idx in row_indices:
                cell = gspread.Cell(row_idx, status_col, scheduled_value)
                cells_to_update.append(cell)
            
            if cells_to_update:
                worksheet.update_cells(cells_to_update)
                return True
            
            return False
            
        except Exception as e:
            st.error(f"❌ Error updating status: {e}")
            return False
    
    def clear_cache(self):
        """Clear the cached data to force a refresh."""
        self.fetch_pending_orders.clear()
