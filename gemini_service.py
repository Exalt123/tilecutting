"""
Gemini AI Service for Tile Cutting Automation
==============================================
Uses Google Gemini to intelligently optimize the production run flow.
Falls back to basic sorting if AI unavailable.
"""

import streamlit as st
import pandas as pd
import json
from typing import Optional

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiOptimizer:
    """Service class for Gemini AI optimization."""
    
    def __init__(self):
        """Initialize the Gemini service with API key from secrets."""
        self.model = None
        self.model_name = None
        self._initialize()
    
    def _initialize(self):
        """Load API key and configure Gemini."""
        if not GEMINI_AVAILABLE:
            return
            
        try:
            api_key = st.secrets.get("gemini_api_key", "")
            
            if not api_key or api_key.startswith("YOUR_"):
                return
            
            genai.configure(api_key=api_key)
            
            # Try different model names (API may support different versions)
            model_options = [
                'gemini-2.0-flash',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro',
            ]
            
            for model_name in model_options:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    break
                except Exception:
                    continue
            
        except Exception as e:
            pass  # Fail silently, will use fallback
    
    def is_ready(self) -> bool:
        """Check if the service is properly configured."""
        return self.model is not None
    
    def optimize_run_flow(self, orders_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Optimize the production run flow.
        
        Uses AI if available, otherwise falls back to basic sorting.
        """
        if orders_df.empty:
            st.warning("⚠️ No orders to optimize")
            return orders_df
        
        # Try AI optimization first
        if self.is_ready():
            result = self._ai_optimize(orders_df)
            if result is not None:
                return result
            st.warning("⚠️ AI optimization failed. Using smart sorting instead.")
        
        # Fallback: basic sorting by cut size
        return self._fallback_optimize(orders_df)
    
    def _fallback_optimize(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Fallback optimization using simple sorting by cut size."""
        df = orders_df.copy()
        
        if 'tile_cut_width' in df.columns and 'tile_cut_length' in df.columns:
            # Sort by cut dimensions to group same sizes together
            df = df.sort_values(['tile_cut_width', 'tile_cut_length'])
            df = df.reset_index(drop=True)
        
        return df
    
    def _ai_optimize(self, orders_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Use Gemini AI to optimize the run flow."""
        try:
            # Prepare orders as JSON for the AI
            cols_for_ai = ['customer', 'job_number', 'tile_width', 'tile_length', 
                          'tile_cut_width', 'tile_cut_length', 'qty_required', 
                          'date_available', '_row_index']
            available_cols = [c for c in cols_for_ai if c in orders_df.columns]
            orders_for_ai = orders_df[available_cols].to_dict('records')
            
            # Build prompt
            prompt = self._build_optimization_prompt(orders_for_ai)
            
            # Call Gemini
            with st.spinner(f"🤖 Optimizing with AI ({self.model_name})..."):
                response = self.model.generate_content(prompt)
            
            # Parse response
            return self._parse_ai_response(response.text, orders_df)
            
        except Exception as e:
            st.error(f"❌ AI Error: {e}")
            return None
    
    def _build_optimization_prompt(self, orders: list) -> str:
        """Build the prompt for Gemini."""
        orders_json = json.dumps(orders, indent=2)
        
        prompt = f"""You are a Production Scheduler for a tile cutting machine.

## TASK
Reorder these tile cutting orders to MINIMIZE machine changeovers.

## RULES
1. Group identical cut sizes together (same tile_cut_width AND tile_cut_length must be consecutive)
2. Keep all data exactly the same - only change the order
3. Include every order exactly once
4. When changing sizes, prefer similar dimensions (e.g., 2×2 → 2×4 better than 2×2 → 12×24)

## INPUT
```json
{orders_json}
```

## OUTPUT
Return ONLY a JSON array with the reordered orders. No explanation, no markdown, just the raw JSON array.
"""
        return prompt
    
    def _parse_ai_response(self, response_text: str, original_df: pd.DataFrame) -> pd.DataFrame:
        """Parse Gemini's response."""
        try:
            # Clean response
            cleaned = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            
            # Parse JSON
            optimized = json.loads(cleaned)
            
            if not isinstance(optimized, list):
                raise ValueError("Response is not a list")
            
            # Convert to DataFrame
            optimized_df = pd.DataFrame(optimized)
            
            # Validate count
            if len(optimized_df) != len(original_df):
                st.warning(f"⚠️ Order count mismatch ({len(optimized_df)} vs {len(original_df)})")
                return None
            
            return optimized_df
            
        except json.JSONDecodeError as e:
            return None
        except Exception as e:
            return None
    
    def calculate_changeovers(self, orders_df: pd.DataFrame) -> int:
        """Count changeovers in the sequence."""
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
        """Generate optimization summary."""
        original_changeovers = self.calculate_changeovers(original_df)
        optimized_changeovers = self.calculate_changeovers(optimized_df)
        changeovers_saved = original_changeovers - optimized_changeovers
        
        unique_sizes = 0
        if not optimized_df.empty and 'tile_cut_width' in optimized_df.columns:
            unique_sizes = optimized_df.groupby(['tile_cut_width', 'tile_cut_length']).ngroups
        
        total_qty = 0
        if 'qty_required' in optimized_df.columns:
            total_qty = optimized_df['qty_required'].sum()
        
        return {
            'original_changeovers': original_changeovers,
            'optimized_changeovers': optimized_changeovers,
            'changeovers_saved': changeovers_saved,
            'time_saved_minutes': changeovers_saved * 5,
            'total_orders': len(optimized_df),
            'unique_sizes': unique_sizes,
            'total_quantity': total_qty
        }
