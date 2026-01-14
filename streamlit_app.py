"""
Technima Flow Optimizer
=======================
A web-based application that optimizes tile cutting production flow
to minimize machine changeovers and downtime.

Author: Exalt Samples LLC
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import services
from services import GoogleSheetsService, GeminiOptimizer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Technima Flow Optimizer",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PASSWORD PROTECTION ---
def check_password() -> bool:
    """Simple password protection for the app."""
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    # Login UI
    st.title("🔷 Technima Flow Optimizer")
    st.markdown("### 🔒 Login Required")
    st.markdown("Enter the shop password to access the production optimizer.")
    
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("🔓 Login", use_container_width=True)
        
        if submit:
            app_password = st.secrets.get("app_password", "Exalt123")
            if password == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
    
    return False


# --- INITIALIZE SERVICES ---
@st.cache_resource
def get_sheets_service():
    """Get cached Google Sheets service instance."""
    return GoogleSheetsService()

@st.cache_resource
def get_gemini_service():
    """Get cached Gemini optimizer instance."""
    return GeminiOptimizer()


# --- MAIN APPLICATION ---
def main():
    """Main application logic."""
    
    # Check authentication
    if not check_password():
        st.stop()
    
    # Initialize services
    sheets_service = get_sheets_service()
    gemini_service = get_gemini_service()
    
    # --- HEADER ---
    st.title("🔷 Technima Flow Optimizer")
    st.markdown("*Minimize changeovers. Maximize production efficiency.*")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.pop('pending_orders', None)
            st.session_state.pop('optimized_orders', None)
            st.rerun()
        
        st.markdown("---")
        
        # Connection status
        st.subheader("📡 Status")
        
        if sheets_service.is_connected():
            st.success("✅ Google Sheets Connected")
        else:
            st.error("❌ Google Sheets Not Connected")
        
        if gemini_service.is_ready():
            st.success("✅ Gemini AI Ready")
        else:
            st.error("❌ Gemini AI Not Configured")
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **Step 1: Refresh Data**
            - Click "Refresh Data" to pull pending orders
            - Only orders NOT marked "Done" or "Scheduled" appear
            
            **Step 2: Optimize**
            - Click "✨ Optimize Run Flow"
            - AI groups orders by cut size
            - Review the optimized sequence
            
            **Step 3: Approve**
            - Click "✅ Approve & Schedule"
            - Orders are marked "Scheduled" in the sheet
            - Ready for production!
            """)
    
    # --- MAIN CONTENT ---
    
    # Initialize session state
    if 'pending_orders' not in st.session_state:
        st.session_state.pending_orders = None
    if 'optimized_orders' not in st.session_state:
        st.session_state.optimized_orders = None
    
    # --- STEP 1: DATA INGESTION ---
    st.header("📥 Step 1: Load Orders")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            sheets_service.clear_cache()
            st.session_state.pending_orders = sheets_service.fetch_pending_orders()
            st.session_state.optimized_orders = None  # Reset optimization
            st.success(f"✅ Loaded {len(st.session_state.pending_orders)} pending orders")
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.pending_orders = None
            st.session_state.optimized_orders = None
            st.rerun()
    
    # Display pending orders (the "Messy List")
    if st.session_state.pending_orders is not None and not st.session_state.pending_orders.empty:
        
        df = st.session_state.pending_orders
        
        # Calculate current changeovers
        current_changeovers = gemini_service.calculate_changeovers(df)
        
        st.markdown("---")
        st.subheader("📋 Current Orders (Unoptimized)")
        
        # Metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Orders", len(df))
        with metric_col2:
            st.metric("Total Quantity", int(df['Quantity'].sum()) if 'Quantity' in df.columns else 0)
        with metric_col3:
            unique_sizes = df.groupby(['Cut Width', 'Cut Height']).ngroups if 'Cut Width' in df.columns else 0
            st.metric("Unique Sizes", unique_sizes)
        with metric_col4:
            st.metric("⚠️ Changeovers", current_changeovers, help="Number of size changes in current order")
        
        # Display table (hide internal columns)
        display_cols = [col for col in df.columns if not col.startswith('_')]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        # --- STEP 2: OPTIMIZATION ---
        st.markdown("---")
        st.header("✨ Step 2: Optimize Run Flow")
        
        if st.button("🤖 Optimize with AI", use_container_width=True, type="primary"):
            optimized = gemini_service.optimize_run_flow(df)
            if optimized is not None:
                st.session_state.optimized_orders = optimized
                st.success("✅ Optimization complete!")
        
        # Display optimized results
        if st.session_state.optimized_orders is not None:
            
            optimized_df = st.session_state.optimized_orders
            summary = gemini_service.get_optimization_summary(df, optimized_df)
            
            st.subheader("🎯 Optimized Run Flow")
            
            # Success metrics
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            
            with result_col1:
                st.metric(
                    "Changeovers (Before)", 
                    summary['original_changeovers']
                )
            with result_col2:
                st.metric(
                    "Changeovers (After)", 
                    summary['optimized_changeovers'],
                    delta=-summary['changeovers_saved'],
                    delta_color="inverse"
                )
            with result_col3:
                st.metric(
                    "🎉 Changeovers Saved", 
                    summary['changeovers_saved'],
                    help="Fewer changeovers = less downtime"
                )
            with result_col4:
                st.metric(
                    "⏱️ Est. Time Saved", 
                    f"{summary['time_saved_minutes']} min",
                    help="Assuming ~5 min per changeover"
                )
            
            # Highlight the improvement
            if summary['changeovers_saved'] > 0:
                st.success(f"🎯 **Optimization reduced changeovers by {summary['changeovers_saved']}!** Estimated time savings: {summary['time_saved_minutes']} minutes.")
            elif summary['changeovers_saved'] == 0:
                st.info("ℹ️ Orders are already optimally grouped by cut size.")
            
            # Display optimized table with size grouping highlighted
            st.markdown("#### 📊 Optimized Sequence")
            
            display_cols_opt = [col for col in optimized_df.columns if not col.startswith('_')]
            
            # Add visual grouping indicator
            optimized_display = optimized_df[display_cols_opt].copy()
            if 'Cut Width' in optimized_display.columns and 'Cut Height' in optimized_display.columns:
                optimized_display['Size Group'] = optimized_display['Cut Width'].astype(str) + '×' + optimized_display['Cut Height'].astype(str)
            
            st.dataframe(
                optimized_display,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # --- STEP 3: APPROVE & SCHEDULE ---
            st.markdown("---")
            st.header("✅ Step 3: Approve & Schedule")
            
            st.warning("⚠️ **This will update the Google Sheet!** All displayed orders will be marked as 'Scheduled'.")
            
            col_approve1, col_approve2 = st.columns([1, 3])
            
            with col_approve1:
                if st.button("✅ Approve & Schedule", use_container_width=True, type="primary"):
                    # Get row indices to update
                    row_indices = optimized_df['_row_index'].tolist()
                    
                    with st.spinner("📝 Updating Google Sheet..."):
                        success = sheets_service.update_status_scheduled(row_indices)
                    
                    if success:
                        st.success(f"✅ Successfully scheduled {len(row_indices)} orders!")
                        st.balloons()
                        
                        # Clear state for next batch
                        st.session_state.pending_orders = None
                        st.session_state.optimized_orders = None
                        sheets_service.clear_cache()
                        
                        st.info("🔄 Click 'Refresh Data' to load the next batch of orders.")
                    else:
                        st.error("❌ Failed to update the sheet. Please try again.")
            
            with col_approve2:
                st.markdown(f"""
                **Ready to schedule {len(optimized_df)} orders:**
                - Total tiles: {summary['total_quantity']:,}
                - Unique sizes: {summary['unique_sizes']}
                - Optimized changeovers: {summary['optimized_changeovers']}
                """)
    
    elif st.session_state.pending_orders is not None and st.session_state.pending_orders.empty:
        st.info("✨ **All caught up!** No pending orders found. All orders are either 'Done' or 'Scheduled'.")
    
    else:
        st.info("👆 Click **Refresh Data** to load pending orders from Google Sheets.")
    
    # --- FOOTER ---
    st.markdown("---")
    st.caption(f"Technima Flow Optimizer v1.0 | © {datetime.now().year} Exalt Samples LLC")


if __name__ == "__main__":
    main()
