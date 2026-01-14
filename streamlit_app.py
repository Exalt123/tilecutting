"""
Tile Cutting Automation
=======================
One-click optimization for tile cutting production flow.
Minimizes changeovers, generates operator-ready PDF.

Author: Exalt Samples LLC
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

# Import services
from gsheets_service import GoogleSheetsService
from gemini_service import GeminiOptimizer
from pdf_generator import generate_run_sheet_pdf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Tile Cutting Automation",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PASSWORD PROTECTION ---
def check_password() -> bool:
    """Simple password protection for the app."""
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.title("🔷 Tile Cutting Automation")
    st.markdown("### 🔒 Login Required")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submit:
                app_password = st.secrets.get("app_password", "Exalt123")
                if password == app_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
    
    return False


# --- INITIALIZE SERVICES ---
@st.cache_resource
def get_sheets_service():
    return GoogleSheetsService()

@st.cache_resource
def get_gemini_service():
    return GeminiOptimizer()


# --- MAIN APPLICATION ---
def main():
    if not check_password():
        st.stop()
    
    sheets_service = get_sheets_service()
    gemini_service = get_gemini_service()
    
    # --- HEADER ---
    col_title, col_logout = st.columns([6, 1])
    with col_title:
        st.title("🔷 Tile Cutting Automation")
        st.caption("Minimize changeovers. Maximize efficiency.")
    with col_logout:
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()
    
    # --- STATUS CHECK ---
    if not sheets_service.is_connected():
        st.error("❌ Google Sheets not connected. Check secrets configuration.")
        st.stop()
    
    if not gemini_service.is_ready():
        st.warning("⚠️ Gemini AI not configured. Optimization will use basic sorting.")
    
    # --- SESSION STATE ---
    if 'optimized_data' not in st.session_state:
        st.session_state.optimized_data = None
    if 'original_changeovers' not in st.session_state:
        st.session_state.original_changeovers = 0
    
    st.markdown("---")
    
    # ========================================
    # STEP 1: RUN OPTIMIZATION (One Click)
    # ========================================
    
    if st.session_state.optimized_data is None:
        st.markdown("## 🚀 Ready to Optimize")
        st.info("Click the button below to pull pending orders and optimize the run sequence.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✨ Run Optimization", use_container_width=True, type="primary"):
                with st.spinner("📥 Loading orders from Google Sheets..."):
                    sheets_service.clear_cache()
                    pending_orders = sheets_service.fetch_pending_orders()
                
                if pending_orders is None or pending_orders.empty:
                    st.success("✅ All caught up! No pending orders to schedule.")
                    st.stop()
                
                # Store original changeovers
                st.session_state.original_changeovers = gemini_service.calculate_changeovers(pending_orders)
                
                with st.spinner("🤖 AI is optimizing the run flow..."):
                    optimized = gemini_service.optimize_run_flow(pending_orders)
                
                if optimized is not None:
                    st.session_state.optimized_data = optimized
                    st.rerun()
                else:
                    st.error("❌ Optimization failed. Please try again.")
    
    else:
        # ========================================
        # STEP 2: REVIEW & EDIT
        # ========================================
        
        df = st.session_state.optimized_data
        
        # Metrics
        current_changeovers = gemini_service.calculate_changeovers(df)
        changeovers_saved = st.session_state.original_changeovers - current_changeovers
        time_saved = changeovers_saved * 5
        total_tiles = int(df['qty_required'].sum()) if 'qty_required' in df.columns else 0
        
        st.markdown("## 📋 Optimized Run Flow")
        
        # Stats row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Orders", len(df))
        col2.metric("Total Tiles", f"{total_tiles:,}")
        col3.metric("Changeovers", current_changeovers, 
                   delta=-changeovers_saved if changeovers_saved > 0 else None,
                   delta_color="inverse")
        col4.metric("Time Saved", f"{time_saved} min" if time_saved > 0 else "—")
        
        unique_sizes = df.groupby(['tile_cut_width', 'tile_cut_length']).ngroups if 'tile_cut_width' in df.columns else 0
        col5.metric("Cut Sizes", unique_sizes)
        
        if changeovers_saved > 0:
            st.success(f"🎯 **Reduced changeovers by {changeovers_saved}!** Est. {time_saved} minutes saved.")
        
        st.markdown("---")
        
        # --- SORTING OPTIONS ---
        st.markdown("### 🔧 Adjust Order")
        
        sort_col1, sort_col2, sort_col3 = st.columns([2, 2, 2])
        
        with sort_col1:
            sort_by = st.selectbox("Sort by", [
                "Current (AI Optimized)",
                "Cut Size (Width × Length)",
                "Cut Width Only",
                "Cut Length Only",
                "Customer",
                "Job Number",
                "Quantity (High to Low)"
            ])
        
        with sort_col2:
            if st.button("↕️ Apply Sort", use_container_width=True):
                if sort_by == "Cut Size (Width × Length)":
                    df = df.sort_values(['tile_cut_width', 'tile_cut_length'])
                elif sort_by == "Cut Width Only":
                    df = df.sort_values('tile_cut_width')
                elif sort_by == "Cut Length Only":
                    df = df.sort_values('tile_cut_length')
                elif sort_by == "Customer":
                    df = df.sort_values('customer')
                elif sort_by == "Job Number":
                    df = df.sort_values('job_number')
                elif sort_by == "Quantity (High to Low)":
                    df = df.sort_values('qty_required', ascending=False)
                
                st.session_state.optimized_data = df.reset_index(drop=True)
                st.rerun()
        
        with sort_col3:
            if st.button("🔄 Re-Optimize with AI", use_container_width=True):
                st.session_state.optimized_data = None
                st.rerun()
        
        st.markdown("---")
        
        # --- EDITABLE TABLE ---
        st.markdown("### 📝 Review & Reorder")
        st.caption("Drag rows to reorder, or edit values directly. Changes are reflected in the final run sheet.")
        
        # Prepare display dataframe
        display_cols = ['customer', 'job_number', 'tile_cut_width', 'tile_cut_length', 
                       'qty_required', 'tile_width', 'tile_length', 'date_available']
        available_cols = [c for c in display_cols if c in df.columns]
        
        # Add a Cut Size column for easy viewing
        display_df = df[available_cols].copy()
        display_df.insert(0, 'Cut Size', 
            df['tile_cut_width'].astype(str) + ' × ' + df['tile_cut_length'].astype(str))
        display_df.insert(0, '#', range(1, len(df) + 1))
        
        # Editable data editor
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "#": st.column_config.NumberColumn("Run #", width="small"),
                "Cut Size": st.column_config.TextColumn("Cut Size", width="medium"),
                "customer": st.column_config.TextColumn("Customer"),
                "job_number": st.column_config.TextColumn("Job #"),
                "tile_cut_width": st.column_config.NumberColumn("Cut W"),
                "tile_cut_length": st.column_config.NumberColumn("Cut L"),
                "qty_required": st.column_config.NumberColumn("Qty"),
                "tile_width": st.column_config.NumberColumn("Tile W"),
                "tile_length": st.column_config.NumberColumn("Tile L"),
                "date_available": st.column_config.TextColumn("Available"),
            },
            height=400
        )
        
        # Update session state if edited
        if not edited_df.equals(display_df):
            # Map edits back to main dataframe
            for col in available_cols:
                if col in edited_df.columns:
                    st.session_state.optimized_data[col] = edited_df[col].values
        
        st.markdown("---")
        
        # ========================================
        # STEP 3: APPROVE & GENERATE PDF
        # ========================================
        
        st.markdown("## ✅ Finalize & Print")
        
        col_approve, col_pdf, col_cancel = st.columns([2, 2, 1])
        
        with col_approve:
            if st.button("📅 Approve & Schedule", use_container_width=True, type="primary"):
                row_indices = st.session_state.optimized_data['_row_index'].tolist()
                
                with st.spinner("📝 Updating Google Sheet..."):
                    success = sheets_service.update_status_scheduled(row_indices)
                
                if success:
                    st.success(f"✅ Scheduled {len(row_indices)} orders!")
                    st.balloons()
                    
                    # Keep data for PDF download, mark as scheduled
                    st.session_state.scheduled = True
                else:
                    st.error("❌ Failed to update sheet. Try again.")
        
        with col_pdf:
            # Generate PDF
            pdf_bytes = generate_run_sheet_pdf(
                st.session_state.optimized_data,
                current_changeovers,
                total_tiles
            )
            
            st.download_button(
                label="📄 Download Run Sheet PDF",
                data=pdf_bytes,
                file_name=f"tile_run_sheet_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col_cancel:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.optimized_data = None
                st.session_state.pop('scheduled', None)
                st.rerun()
        
        # Show scheduled confirmation
        if st.session_state.get('scheduled'):
            st.info("✅ Orders have been scheduled. Download the PDF for the operator, then click **Clear** to start a new batch.")
    
    # --- FOOTER ---
    st.markdown("---")
    st.caption(f"Tile Cutting Automation v1.0 | © {datetime.now().year} Exalt Samples LLC")


if __name__ == "__main__":
    main()
