"""
PDF Generator for Tile Cutting Automation
==========================================
Generates clean, operator-ready run sheets.
"""

from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import pandas as pd


def generate_run_sheet_pdf(df: pd.DataFrame, changeovers: int, total_tiles: int) -> bytes:
    """
    Generate a clean PDF run sheet for the operator.
    
    Args:
        df: Optimized DataFrame with orders
        changeovers: Number of changeovers in the sequence
        total_tiles: Total quantity of tiles
        
    Returns:
        bytes: PDF file as bytes
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.5*inch,
        rightMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    styles = getSampleStyleSheet()
    elements = []
    
    # --- TITLE ---
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor('#1f4788')
    )
    elements.append(Paragraph("🔷 TILE CUTTING RUN SHEET", title_style))
    
    # --- DATE & SUMMARY ---
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=10
    )
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        date_style
    ))
    elements.append(Spacer(1, 0.2*inch))
    
    # --- SUMMARY BOX ---
    summary_data = [
        ['Total Orders', 'Total Tiles', 'Changeovers', 'Unique Sizes'],
        [
            str(len(df)),
            f"{total_tiles:,}",
            str(changeovers),
            str(df.groupby(['tile_cut_width', 'tile_cut_length']).ngroups) if 'tile_cut_width' in df.columns else '—'
        ]
    ]
    
    summary_table = Table(summary_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch, 1.8*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#e8f4f8')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, 1), 16),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1f4788'))
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # --- RUN SEQUENCE TABLE ---
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.HexColor('#1f4788')
    )
    elements.append(Paragraph("RUN SEQUENCE (Optimized)", section_style))
    
    # Build table data
    header_style = ParagraphStyle('Header', fontSize=9, alignment=TA_CENTER, textColor=colors.white, fontName='Helvetica-Bold')
    cell_style = ParagraphStyle('Cell', fontSize=9, alignment=TA_CENTER, leading=11)
    
    table_data = [[
        Paragraph('#', header_style),
        Paragraph('Cut Size', header_style),
        Paragraph('Customer', header_style),
        Paragraph('Job #', header_style),
        Paragraph('Qty', header_style),
        Paragraph('Tile Size', header_style),
        Paragraph('✓', header_style)
    ]]
    
    current_size = None
    for idx, row in df.iterrows():
        cut_size = f"{row.get('tile_cut_width', '')} × {row.get('tile_cut_length', '')}"
        tile_size = f"{row.get('tile_width', '')} × {row.get('tile_length', '')}"
        
        # Highlight size changes
        is_new_size = cut_size != current_size
        current_size = cut_size
        
        row_data = [
            Paragraph(str(idx + 1), cell_style),
            Paragraph(f"<b>{cut_size}</b>" if is_new_size else cut_size, cell_style),
            Paragraph(str(row.get('customer', '')), cell_style),
            Paragraph(str(row.get('job_number', '')), cell_style),
            Paragraph(str(int(row.get('qty_required', 0))), cell_style),
            Paragraph(tile_size, cell_style),
            Paragraph('☐', cell_style)  # Checkbox for operator
        ]
        table_data.append(row_data)
    
    # Create table
    col_widths = [0.4*inch, 1.0*inch, 1.5*inch, 1.0*inch, 0.6*inch, 1.0*inch, 0.4*inch]
    run_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Style with alternating rows
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    
    # Alternating row colors
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f5f5f5')))
    
    # Highlight changeover rows (where size changes)
    current_size = None
    for i, row in enumerate(df.iterrows()):
        idx, data = row
        cut_size = f"{data.get('tile_cut_width', '')} × {data.get('tile_cut_length', '')}"
        if current_size is not None and cut_size != current_size:
            table_style.append(('LINEABOVE', (0, i + 1), (-1, i + 1), 2, colors.HexColor('#ff6b6b')))
        current_size = cut_size
    
    run_table.setStyle(TableStyle(table_style))
    elements.append(run_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # --- OPERATOR SIGN-OFF ---
    elements.append(Paragraph("OPERATOR SIGN-OFF", section_style))
    
    signoff_style = ParagraphStyle('Signoff', fontSize=11, leading=20, spaceAfter=5)
    elements.append(Paragraph("<b>Operator Name:</b> _________________________________", signoff_style))
    elements.append(Paragraph("<b>Date Completed:</b> _________________________________", signoff_style))
    elements.append(Paragraph("<b>Start Time:</b> _____________ <b>End Time:</b> _____________", signoff_style))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("<b>Notes:</b>", signoff_style))
    elements.append(Paragraph("_______________________________________________________________", signoff_style))
    elements.append(Paragraph("_______________________________________________________________", signoff_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
