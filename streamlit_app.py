import streamlit as st
import math
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# Page config
st.set_page_config(
    page_title="MDF Cutting Optimizer",
    page_icon="🪵",
    layout="wide"
)

# --- HELPER FUNCTIONS ---
def safe_float(value, default=0.0):
    """Safely convert value to float, return default if conversion fails."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace('$', '').replace(',', '')
        if value == "":
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_google_sheets_client(credentials_json):
    """Initialize Google Sheets client from service account credentials."""
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(credentials_json, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_mdf_sheet_data(client, spreadsheet_id, sheet_name_or_id):
    """Fetch MDF material data from a specific sheet in the spreadsheet."""
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Try to get sheet by name or ID
        try:
            if isinstance(sheet_name_or_id, int) or (isinstance(sheet_name_or_id, str) and sheet_name_or_id.isdigit()):
                sheet = spreadsheet.get_worksheet_by_id(int(sheet_name_or_id))
            else:
                # Remove 'gid=' prefix if present
                if sheet_name_or_id.startswith('gid='):
                    sheet_name_or_id = sheet_name_or_id.replace('gid=', '')
                if sheet_name_or_id.isdigit():
                    sheet = spreadsheet.get_worksheet_by_id(int(sheet_name_or_id))
                else:
                    sheet = spreadsheet.worksheet(sheet_name_or_id)
        except:
            # If sheet name doesn't work, try getting all sheets and finding by ID
            if isinstance(sheet_name_or_id, int) or (isinstance(sheet_name_or_id, str) and sheet_name_or_id.replace('gid=', '').isdigit()):
                sheet_id_str = str(sheet_name_or_id).replace('gid=', '')
                all_sheets = spreadsheet.worksheets()
                for s in all_sheets:
                    if str(s.id) == sheet_id_str:
                        sheet = s
                        break
                else:
                    raise ValueError(f"Sheet ID {sheet_name_or_id} not found")
            else:
                raise
        
        # Get all records
        records = sheet.get_all_records()
        
        # Convert to list of dicts with lowercase keys
        result = []
        for record in records:
            clean_row = {k.strip().lower(): v for k, v in record.items()}
            result.append(clean_row)
        
        return result
    except Exception as e:
        st.error(f"Error fetching MDF sheet data: {e}")
        return []

# --- MDF CALCULATION FUNCTION ---
def calculate_mdf_options(row, job_w, job_h, qty, target_thickness=None):
    """Calculate MDF cost for a single material row.
    
    Returns None if material doesn't fit or doesn't match filters.
    """
    # Get sheet dimensions
    sheet_w_val = row.get('sheet_width') or row.get('sheet_w')
    if not sheet_w_val:
        for key in row.keys():
            if key.lower() in ['sheet_width', 'sheet_w', 'width']:
                sheet_w_val = row.get(key)
                break
    sheet_w = safe_float(sheet_w_val, None)
    
    sheet_h_val = row.get('sheet_length') or row.get('sheet_h') or row.get('length')
    if not sheet_h_val:
        for key in row.keys():
            if key.lower() in ['sheet_length', 'sheet_h', 'length', 'sheet_length']:
                sheet_h_val = row.get(key)
                break
    sheet_h = safe_float(sheet_h_val, None)
    
    # Skip if no dimensions
    if sheet_w is None or sheet_h is None or sheet_w == 0 or sheet_h == 0:
        return None
    
    # Filter by thickness if specified
    if target_thickness is not None and target_thickness != "":
        row_thickness = safe_float(row.get('thickness'), None)
        target_thickness_float = safe_float(target_thickness, None)
        if row_thickness is None or target_thickness_float is None:
            return None
        # Allow small floating point differences (0.01 tolerance)
        if abs(row_thickness - target_thickness_float) > 0.01:
            return None
    
    # Get cost
    cost_val = row.get('cost_per_sheet') or row.get('cost') or row.get('costpersheet')
    if not cost_val:
        for key in row.keys():
            if key.lower() in ['cost_per_sheet', 'cost', 'costpersheet', 'sheet_cost']:
                cost_val = row.get(key)
                break
    cost = safe_float(cost_val, 0)
    
    # Get material name from 'sku' column, fallback to material_name
    name_val = row.get('sku') or row.get('material_name') or row.get('material')
    if not name_val:
        for key in row.keys():
            if key.lower() in ['sku', 'material_name', 'material', 'name']:
                name_val = row.get(key)
                break
    name = name_val or 'Unknown MDF'
    
    # Check if in stock (optional filter)
    in_stock_raw = row.get('in_stock') or row.get('instock')
    in_stock_str = str(in_stock_raw).strip().upper() if in_stock_raw is not None else ''
    is_in_stock = in_stock_str in ['Y', 'YES', 'TRUE', '1', 'IN STOCK', '']
    
    # Get thickness for display
    row_thickness = safe_float(row.get('thickness'), None)
    
    # Calculate yield using original formula
    saw_kerf = 0.25  # Standard saw kerf
    eff_w = job_w + saw_kerf
    eff_h = job_h + saw_kerf
    
    # Check if job fits in either orientation
    if (eff_w > sheet_w and eff_w > sheet_h) or (eff_h > sheet_w and eff_h > sheet_h):
        return None  # Doesn't fit
    
    # Calculate yield for both orientations
    yield_norm = math.floor(sheet_w / eff_w) * math.floor(sheet_h / eff_h)
    yield_rot = math.floor(sheet_w / eff_h) * math.floor(sheet_h / eff_w)
    parts_per_sheet = max(yield_norm, yield_rot)
    orientation = "Normal" if yield_norm >= yield_rot else "Rotated"
    
    if parts_per_sheet == 0:
        return None
    
    # Calculate sheets needed
    sheets_needed = math.ceil(qty / parts_per_sheet)
    total_cost = sheets_needed * cost
    
    # Calculate waste percentage
    waste = 0
    if sheets_needed > 0:
        area_used = qty * (job_w * job_h)
        area_total = sheets_needed * (sheet_w * sheet_h)
        if area_total > 0:
            waste = 100 - ((area_used / area_total) * 100)
    
    # Calculate efficiency (how many pieces we'll actually cut vs what we need)
    pieces_cut = sheets_needed * parts_per_sheet
    overage = pieces_cut - qty
    
    return {
        "material": name,
        "sku": row.get('sku', name),
        "thickness": row_thickness,
        "sheet_size": f"{sheet_w} x {sheet_h}",
        "sheet_width": sheet_w,
        "sheet_height": sheet_h,
        "cost_per_sheet": cost,
        "sheets_needed": sheets_needed,
        "parts_per_sheet": parts_per_sheet,
        "orientation": orientation,
        "total_cost": round(total_cost, 2),
        "waste_pct": round(waste, 1),
        "pieces_cut": pieces_cut,
        "overage": overage,
        "in_stock": is_in_stock
    }

# --- MAIN APP ---
st.title("🪵 MDF Cutting Optimizer")
st.markdown("Optimize your cut list by finding the most efficient and cost-effective sheet materials for your cutting needs.")

# Sidebar for Google Sheets configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Help section
    with st.expander("❓ How to Get JSON Credentials", expanded=False):
        st.markdown("""
        **Quick Steps:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create/select a project
        3. Enable **Google Sheets API** and **Google Drive API**
        4. Go to **IAM & Admin > Service Accounts**
        5. Create a service account
        6. Go to **Keys** tab, click **Add Key > Create new key > JSON**
        7. Download the JSON file
        8. Copy the service account **email**
        9. Share your Google Sheet with that email (Viewer access)
        10. Get your **Spreadsheet ID** from the URL
        
        📖 **Detailed guide**: See `GOOGLE_SHEETS_SETUP.md` for step-by-step instructions with screenshots.
        """)
    
    # Google Sheets credentials
    st.subheader("Google Sheets Access")
    credentials_method = st.radio(
        "Credentials Method",
        ["Upload JSON", "Paste JSON"],
        help="Upload a service account JSON file or paste the JSON content"
    )
    
    credentials_json = None
    if credentials_method == "Upload JSON":
        cred_file = st.file_uploader("Service Account JSON", type=["json"], key="cred_upload")
        if cred_file:
            import json
            credentials_json = json.load(cred_file)
    else:
        cred_text = st.text_area("Service Account JSON", height=200, help="Paste your service account JSON credentials", key="cred_paste")
        if cred_text:
            import json
            try:
                credentials_json = json.loads(cred_text)
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    if credentials_json:
        st.success("✓ Credentials loaded")
        
        # Spreadsheet configuration
        st.subheader("Spreadsheet Settings")
        
        with st.expander("📍 How to Find Spreadsheet ID"):
            st.markdown("""
            **In your Google Sheet URL:**
            ```
            https://docs.google.com/spreadsheets/d/SPREADSHEET_ID_HERE/edit
            ```
            Copy the long string between `/d/` and `/edit`
            """)
        
        spreadsheet_id = st.text_input(
            "Spreadsheet ID",
            value="13wQEWzY7oxtHQX6wAeXn2G6f9Tp4fVBiYBA7Tw2u5aA",
            help="The ID from your Google Sheets URL (between /d/ and /edit)"
        )
        
        with st.expander("📍 How to Find Sheet Name/ID"):
            st.markdown("""
            **Option 1 - Sheet Name (Easiest):**
            - Look at the bottom tabs of your sheet
            - Use the tab name (e.g., "Materials", "Sheet1")
            
            **Option 2 - Sheet ID (GID):**
            - Click on the sheet tab you want
            - Look at the URL for `#gid=123456789`
            - Use just the number (e.g., `637696834`) or with prefix (e.g., `gid=637696834`)
            """)
        
        mdf_sheet = st.text_input(
            "MDF Sheet Name/ID", 
            value="gid=0", 
            help="Sheet tab name (e.g., 'Materials') or ID number (e.g., '637696834')"
        )
        
        # Test connection button
        if st.button("🔗 Test Connection"):
            client = get_google_sheets_client(credentials_json)
            if client:
                try:
                    spreadsheet = client.open_by_key(spreadsheet_id)
                    st.success(f"✓ Connected to: {spreadsheet.title}")
                    mdf_data = fetch_mdf_sheet_data(client, spreadsheet_id, mdf_sheet)
                    if mdf_data:
                        st.info(f"✓ Found {len(mdf_data)} material rows")
                    else:
                        st.warning("⚠ No data found in MDF sheet")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# Main form
st.header("📐 Cut List Parameters")

with st.form("cut_list_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        width = st.number_input("Width (inches)", min_value=0.1, value=14.0, step=0.1, format="%.2f")
        height = st.number_input("Height (inches)", min_value=0.1, value=16.0, step=0.1, format="%.2f")
    
    with col2:
        qty = st.number_input("Quantity Needed", min_value=1, value=1000, step=1)
        thickness = st.text_input("Thickness (optional)", value="", help="Filter materials by thickness (e.g., 0.5, 0.75, 1.0)")
    
    with col3:
        show_all = st.checkbox("Show All Options", value=False, help="Show all material options, not just the best")
        filter_in_stock = st.checkbox("Only Show In-Stock", value=True, help="Only show materials marked as in stock")
    
    submitted = st.form_submit_button("🔍 Find Optimal Materials", use_container_width=True)

# Process calculation when form is submitted
if submitted:
    if not credentials_json:
        st.warning("⚠️ Please configure Google Sheets credentials in the sidebar first.")
        st.stop()
    
    if not width or not height or not qty:
        st.error("Please fill in width, height, and quantity.")
        st.stop()
    
    with st.spinner("Fetching materials from Google Sheets..."):
        # Initialize client
        client = get_google_sheets_client(credentials_json)
        
        if not client:
            st.error("Failed to connect to Google Sheets. Please check your credentials.")
            st.stop()
        
        # Fetch MDF data
        mdf_rows = fetch_mdf_sheet_data(client, spreadsheet_id, mdf_sheet)
        
        if not mdf_rows:
            st.error("No data found in MDF sheet. Please check your sheet ID and permissions.")
            st.stop()
    
    # Calculate options for all materials
    with st.spinner("Calculating optimal materials..."):
        mdf_results = []
        for row in mdf_rows:
            result = calculate_mdf_options(row, width, height, qty, thickness if thickness else None)
            if result:
                # Filter by in-stock if requested
                if filter_in_stock and not result['in_stock']:
                    continue
                mdf_results.append(result)
    
    if not mdf_results:
        st.error("❌ No materials found that fit your requirements.")
        st.info("Try adjusting:")
        st.markdown("- Remove thickness filter")
        st.markdown("- Uncheck 'Only Show In-Stock'")
        st.markdown("- Check that your sheet has materials with appropriate dimensions")
        st.stop()
    
    # Sort by total cost (ascending) - best is cheapest
    mdf_results.sort(key=lambda x: x['total_cost'])
    
    # Display results
    st.header("📊 Optimization Results")
    
    # Summary metrics
    best_option = mdf_results[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Best Material", best_option['material'])
    with col2:
        st.metric("Total Cost", f"${best_option['total_cost']:.2f}")
    with col3:
        st.metric("Sheets Needed", best_option['sheets_needed'])
    with col4:
        st.metric("Parts per Sheet", best_option['parts_per_sheet'])
    with col5:
        st.metric("Waste", f"{best_option['waste_pct']:.1f}%")
    
    # Detailed results table
    st.subheader("All Options" if show_all else f"Best Option (showing 1 of {len(mdf_results)} options)")
    
    # Prepare data for display
    display_results = mdf_results if show_all else [best_option]
    
    # Create DataFrame
    df_data = []
    for result in display_results:
        df_data.append({
            "Rank": mdf_results.index(result) + 1,
            "Material (SKU)": f"{result['material']} ({result['sku']})" if result['sku'] else result['material'],
            "Thickness": f"{result['thickness']}\"" if result['thickness'] else "N/A",
            "Sheet Size": result['sheet_size'],
            "Cost/Sheet": f"${result['cost_per_sheet']:.2f}",
            "Sheets Needed": result['sheets_needed'],
            "Parts/Sheet": result['parts_per_sheet'],
            "Orientation": result['orientation'],
            "Total Cost": f"${result['total_cost']:.2f}",
            "Waste %": f"{result['waste_pct']:.1f}%",
            "Pieces Cut": result['pieces_cut'],
            "Overage": result['overage'],
            "In Stock": "✓" if result['in_stock'] else "✗"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed breakdown for best option
    st.subheader("📋 Detailed Breakdown - Best Option")
    
    breakdown_col1, breakdown_col2 = st.columns(2)
    
    with breakdown_col1:
        st.markdown("**Material Information**")
        st.json({
            "Material": best_option['material'],
            "SKU": best_option['sku'],
            "Thickness": f"{best_option['thickness']}\"" if best_option['thickness'] else "N/A",
            "Sheet Size": best_option['sheet_size'],
            "Cost per Sheet": f"${best_option['cost_per_sheet']:.2f}"
        })
        
        st.markdown("**Cutting Details**")
        st.json({
            "Pieces Needed": qty,
            "Pieces per Sheet": best_option['parts_per_sheet'],
            "Sheets Needed": best_option['sheets_needed'],
            "Total Pieces Cut": best_option['pieces_cut'],
            "Overage (extra pieces)": best_option['overage'],
            "Orientation": best_option['orientation']
        })
    
    with breakdown_col2:
        st.markdown("**Cost Breakdown**")
        st.json({
            "Sheets Needed": best_option['sheets_needed'],
            "Cost per Sheet": f"${best_option['cost_per_sheet']:.2f}",
            "Total Material Cost": f"${best_option['total_cost']:.2f}",
            "Cost per Piece": f"${best_option['total_cost'] / qty:.4f}" if qty > 0 else "$0.0000"
        })
        
        st.markdown("**Efficiency Metrics**")
        st.json({
            "Material Utilization": f"{100 - best_option['waste_pct']:.1f}%",
            "Waste": f"{best_option['waste_pct']:.1f}%",
            "Yield per Sheet": best_option['parts_per_sheet'],
            "Sheets per 1000 Pieces": round((best_option['sheets_needed'] / qty) * 1000, 2) if qty > 0 else 0
        })
    
    # Show toggle to see all options
    if not show_all and len(mdf_results) > 1:
        st.info(f"💡 Found {len(mdf_results)} total options. Check 'Show All Options' above to see all materials sorted by cost.")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### Setup (First Time Only)
    1. **Get Google Sheets Service Account**:
       - Go to [Google Cloud Console](https://console.cloud.google.com/)
       - Create a project or select existing one
       - Enable Google Sheets API and Google Drive API
       - Create a Service Account and download JSON key
       - Share your Google Sheet with the service account email (found in JSON)
    
    2. **Configure in Sidebar**:
       - Upload or paste your service account JSON
       - Enter your Spreadsheet ID (from the URL)
       - Enter your MDF sheet name or ID (e.g., "Materials" or "637696834")
       - Click "Test Connection" to verify
    
    ### Using the Tool
    1. **Enter Cut List Parameters**:
       - Width and Height of pieces to cut (in inches)
       - Quantity needed
       - Thickness filter (optional) - will only show matching thickness materials
       - Toggle options to show all materials or filter by in-stock status
    
    2. **Click "Find Optimal Materials"**
    
    3. **Review Results**:
       - Best material option is highlighted
       - See cost, waste, and efficiency metrics
       - Toggle "Show All Options" to see all materials sorted by cost
    """)

# Footer
st.markdown("---")
st.caption("💡 Tip: This tool calculates the most cost-effective sheet size based on yield, waste percentage, and material cost.")
