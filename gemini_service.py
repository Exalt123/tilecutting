"""
Gemini AI Service for Tile Cutting Automation
==============================================
Uses Google Gemini 1.5 Flash to intelligently optimize the production run flow.
"""

import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
from typing import Optional


class GeminiOptimizer:
    """Service class for Gemini AI optimization."""
    
    def __init__(self):
        """Initialize the Gemini service with API key from secrets."""
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Load API key and configure Gemini."""
        try:
            api_key = st.secrets.get("gemini_api_key", "")
            
            if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
                st.error("❌ Gemini API key not configured in secrets.")
                return
            
            genai.configure(api_key=api_key)
            
            # Use Gemini 1.5 Flash for speed and cost efficiency
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini: {e}")
    
    def is_ready(self) -> bool:
        """Check if the service is properly configured."""
        return self.model is not None
    
    def optimize_run_flow(self, orders_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Use Gemini AI to optimize the production run flow.
        
        The AI acts as a Production Planner with these priorities:
        1. HARD CONSTRAINT: Group identical cut sizes together (tile_cut_width × tile_cut_length)
        2. SOFT CONSTRAINT: Transition to similar sizes (minimize changeover impact)
        
        Args:
            orders_df: DataFrame with pending orders
            
        Returns:
            pd.DataFrame: Reordered DataFrame optimized for production, or None on error
        """
        if not self.is_ready():
            st.error("❌ Gemini AI not configured")
            return None
        
        if orders_df.empty:
            st.warning("⚠️ No orders to optimize")
            return orders_df
        
        try:
            # Prepare orders as JSON for the AI
            cols_for_ai = ['customer', 'job_number', 'tile_width', 'tile_length', 
                          'tile_cut_width', 'tile_cut_length', 'qty_required', 
                          'date_available', '_row_index']
            available_cols = [c for c in cols_for_ai if c in orders_df.columns]
            orders_for_ai = orders_df[available_cols].to_dict('records')
            
            # Craft the prompt for Gemini
            prompt = self._build_optimization_prompt(orders_for_ai)
            
            # Call Gemini
            with st.spinner("🤖 AI is optimizing the run flow..."):
                response = self.model.generate_content(prompt)
            
            # Parse the response
            optimized_orders = self._parse_ai_response(response.text, orders_df)
            
            return optimized_orders
            
        except Exception as e:
            st.error(f"❌ Optimization failed: {e}")
            return None
    
    def _build_optimization_prompt(self, orders: list) -> str:
        """
        Build the prompt that instructs Gemini to act as a production scheduler.
        
        Args:
            orders: List of order dictionaries
            
        Returns:
            str: The complete prompt for Gemini
        """
        orders_json = json.dumps(orders, indent=2)
        
        prompt = f"""You are an expert Production Scheduler for a tile cutting machine.

## YOUR MISSION
Reorder the following list of tile cutting orders to MINIMIZE machine changeovers and downtime.

## HARD CONSTRAINTS (MUST FOLLOW)
1. **Group identical cut sizes together** - All orders with the same (tile_cut_width × tile_cut_length) MUST be consecutive
2. **Preserve all order data** - Do not modify any values, only reorder the list
3. **Include ALL orders** - Every order in the input must appear exactly once in the output

## SOFT CONSTRAINTS (OPTIMIZE FOR)
1. **Smooth transitions** - When changing cut sizes, prefer going to similar dimensions
   - Example: 2×2 → 2×3 is better than 2×2 → 24×48
2. **Width changes first** - If possible, change one dimension at a time
3. **Small to large** - Generally prefer ordering from smaller cuts to larger cuts within similar groups

## KEY FIELDS
- **tile_cut_width**: The CUT width (what we're cutting TO) - PRIMARY grouping key
- **tile_cut_length**: The CUT length (what we're cutting TO) - PRIMARY grouping key
- tile_width/tile_length: Original tile dimensions (less important for scheduling)

## INPUT ORDERS (JSON)
```json
{orders_json}
```

## REQUIRED OUTPUT FORMAT
Return ONLY a valid JSON array with the reordered orders. Include ALL original fields.
Do not include any explanation, markdown formatting, or code blocks - just the raw JSON array.

Example output format:
[{{"customer": "ABC", "job_number": "123", "tile_cut_width": 2, "tile_cut_length": 2, ...}}, ...]
"""
        return prompt
    
    def _parse_ai_response(self, response_text: str, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse Gemini's response and convert back to DataFrame.
        
        Args:
            response_text: Raw response from Gemini
            original_df: Original DataFrame for fallback/validation
            
        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        try:
            # Clean the response (remove markdown code blocks if present)
            cleaned_response = response_text.strip()
            
            # Remove markdown code blocks
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                # Remove first and last lines (```json and ```)
                cleaned_response = "\n".join(lines[1:-1])
            
            # Parse JSON
            optimized_orders = json.loads(cleaned_response)
            
            if not isinstance(optimized_orders, list):
                raise ValueError("AI response is not a list")
            
            # Convert to DataFrame
            optimized_df = pd.DataFrame(optimized_orders)
            
            # Validate: ensure we have all orders
            if len(optimized_df) != len(original_df):
                st.warning(f"⚠️ Order count mismatch. Expected {len(original_df)}, got {len(optimized_df)}. Using original order.")
                return original_df
            
            # Add optimization metadata
            optimized_df['_optimized'] = True
            
            return optimized_df
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse AI response: {e}")
            st.code(response_text[:500], language="text")
            return original_df
        except Exception as e:
            st.error(f"❌ Error processing AI response: {e}")
            return original_df
    
    def calculate_changeovers(self, orders_df: pd.DataFrame) -> int:
        """
        Calculate the number of changeovers in the current order sequence.
        
        A changeover occurs when tile_cut_width or tile_cut_length changes between consecutive rows.
        
        Args:
            orders_df: DataFrame with orders
            
        Returns:
            int: Number of changeovers
        """
        if orders_df.empty or len(orders_df) < 2:
            return 0
        
        changeovers = 0
        prev_width = None
        prev_length = None
        
        for _, row in orders_df.iterrows():
            curr_width = row.get('tile_cut_width', 0)
            curr_length = row.get('tile_cut_length', 0)
            
            if prev_width is not None:
                if curr_width != prev_width or curr_length != prev_length:
                    changeovers += 1
            
            prev_width = curr_width
            prev_length = curr_length
        
        return changeovers
    
    def get_optimization_summary(self, original_df: pd.DataFrame, optimized_df: pd.DataFrame) -> dict:
        """
        Generate a summary comparing original vs optimized run flow.
        
        Args:
            original_df: Original order DataFrame
            optimized_df: Optimized order DataFrame
            
        Returns:
            dict: Summary statistics
        """
        original_changeovers = self.calculate_changeovers(original_df)
        optimized_changeovers = self.calculate_changeovers(optimized_df)
        
        changeovers_saved = original_changeovers - optimized_changeovers
        
        # Estimate time saved (assume ~5 minutes per changeover)
        time_saved_minutes = changeovers_saved * 5
        
        # Count unique cut sizes
        unique_sizes = 0
        if not optimized_df.empty and 'tile_cut_width' in optimized_df.columns:
            unique_sizes = optimized_df.groupby(['tile_cut_width', 'tile_cut_length']).ngroups
        
        # Total quantity
        total_qty = 0
        if 'qty_required' in optimized_df.columns:
            total_qty = optimized_df['qty_required'].sum()
        
        return {
            'original_changeovers': original_changeovers,
            'optimized_changeovers': optimized_changeovers,
            'changeovers_saved': changeovers_saved,
            'time_saved_minutes': time_saved_minutes,
            'total_orders': len(optimized_df),
            'unique_sizes': unique_sizes,
            'total_quantity': total_qty
        }
