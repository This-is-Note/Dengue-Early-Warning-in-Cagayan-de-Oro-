import streamlit as st
import pandas as pd
import numpy as np
import base64
# Removed: import plotly.express as px

# --- Configuration and Setup (Run only once) ---

# Set a wide layout and a professional title/icon
st.set_page_config(
    page_title="Dengue Risk Predictor",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a cleaner, modern dark look (Same as before)
st.markdown("""
<style>
    /* Main Content Styling - Subtle Gradient and Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
    
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); /* Deep blue/slate dark gradient */
        color: #e2e8f0; /* Light text for contrast */
    }

    /* Header styling */
    h1 {
        color: #60a5fa; /* Vibrant blue for main title */
        border-bottom: 3px solid #60a5fa;
        padding-bottom: 10px;
        text-shadow: 0 0 8px rgba(96, 165, 250, 0.5); /* Subtle title glow */
        font-weight: 700;
    }

    /* Subheader styling */
    h2, h3 {
        color: #93c5fd; /* Lighter blue for subheaders */
        border-left: 5px solid #60a5fa;
        padding-left: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
        font-weight: 600;
    }

    /* Custom containers for input and output cards - Added Subtle Hover */
    .stContainer {
        border: 1px solid #334155; /* Slate border for cards */
        border-radius: 12px;
        padding: 25px;
        background-color: #1f2937; /* Slightly lighter dark background for cards */
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); 
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .stContainer:hover {
        box-shadow: 0 0 25px rgba(96, 165, 250, 0.15); /* Blue glow on hover */
    }

    /* Input field background in dark mode */
    .stNumberInput div[data-baseweb="input"] input, 
    .stSelectbox div[data-baseweb="select"] div[role="button"],
    .stRadio div[data-baseweb="radio"] label {
        background-color: #0f172a; /* Deepest dark for input fields */
        color: #e2e8f0;
        border-radius: 6px;
        border: 1px solid #334155;
    }
    /* Fixing the radio button background */
    .stRadio > label {
        background-color: transparent !important;
    }

    /* Button Styling - Enhanced Hover */
    .stButton>button {
        background-color: #60a5fa; /* Primary blue button */
        color: white;
        border-radius: 8px;
        padding: 12px 25px;
        font-weight: 700;
        transition: all 0.2s ease-in-out;
        border: none;
        box-shadow: 0 4px 10px rgba(96, 165, 250, 0.3);
    }
    .stButton>button:hover {
        background-color: #3b82f6; /* Darker blue on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(96, 165, 250, 0.5); 
    }
    
    /* Risk Decision Card Styling */
    .decision-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 36px;
        font-weight: 900; /* Extra bold for the decision */
        color: white; 
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
        margin-top: 15px;
        margin-bottom: 20px; 
    }
    
    /* Probability Card Styling */
    .probability-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: 500;
        color: white; 
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        margin-top: 10px;
    }

    /* Color classes */
    .low-risk-color { background-color: #10b981; } /* Emerald Green */
    .moderate-risk-color { background-color: #fbbf24; } /* Amber Yellow */
    .high-risk-color { background-color: #ef4444; } /* Red */


    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #60a5fa !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

@st.cache_data
def load_data():
    """Load, clean, and pre-process the dengue data."""
    try:
        # Assuming the CSV is accessible via its filename in this environment
        df = pd.read_csv('doh-epi-dengue-data-2016-2021.csv', skiprows=[1])
        df.columns = ['loc', 'cases', 'deaths', 'date', 'Region']
        
        # Clean data types
        df['cases'] = pd.to_numeric(df['cases'], errors='coerce', downcast='integer')
        df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce', downcast='integer')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with NaT dates or NaNs in cases after cleaning
        df.dropna(subset=['date', 'cases'], inplace=True)
        
        # Add temporal features for analysis
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%b')
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Group by date for weekly national trend chart 
        df_weekly = df.groupby('date')[['cases', 'deaths']].sum().reset_index()
        df_weekly['year'] = df_weekly['date'].dt.year
        df_weekly['month'] = df_weekly['date'].dt.month
        
        return df, df_weekly

    except FileNotFoundError:
        st.error("Data file 'doh-epi-dengue-data-2016-2021.csv' not found. Please ensure it is uploaded.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame(), pd.DataFrame()


# --- UPDATED Model Stub/Simulation ---

class MockModel:
    """
    Simulates the scikit-learn model's predict_proba method.
    Updated to ONLY use temporal features (simulated_week_of_year) for risk prediction.
    It expects all four features in the input DataFrame but ignores the case counts
    to adhere to the new user input requirement.
    """
    def predict_proba(self, X):
        # Features are now expected to include the old lagged case columns 
        # but the logic only uses the temporal one.
        simulated_week_of_year = X['simulated_week_of_year'].iloc[0] 

        # Seasonal risk factor (peak in mid-year, around week 30)
        seasonal_peak_center = 30 
        # Base risk level is 0.1, max seasonal risk is 0.8
        seasonal_factor = np.exp(-0.5 * ((simulated_week_of_year - seasonal_peak_center) / 15)**2)
        raw_prob = 0.1 + (seasonal_factor * 0.7) # Risk from 10% to 80% based on season
        
        # Add a small random jitter to make it feel less deterministic
        raw_prob = np.clip(raw_prob + np.random.uniform(-0.03, 0.03), 0.01, 0.99)

        # Return the required format: [[Prob_Class_0 (Low Risk), Prob_Class_1 (High Risk)]]
        return [[1.0 - raw_prob, raw_prob]]


loaded_model = MockModel()

def get_risk_style(probability):
    """Determines the risk level, returns the classification, icon, and bar color."""
    if probability < 30:
        return "Low Risk", "🛡️", "low-risk-color"
    elif probability < 60:
        return "Moderate Risk", "⚠️", "moderate-risk-color"
    else:
        return "High Risk", "🛑", "high-risk-color"

def predict_case_range(c_simulated_week_of_year):
    """
    SIMULATION: Calculates a plausible case count and range based only on seasonality.
    """
    seasonal_peak_center = 30
    base_cases = 50 
    max_increase = 200 
    
    # Gaussian-like seasonal factor (0 to 1)
    seasonal_factor = np.exp(-0.5 * ((c_simulated_week_of_year - seasonal_peak_center) / 15)**2)
    
    # Base prediction
    base_prediction = base_cases + (max_increase * seasonal_factor)
    
    # Add a little randomness
    base_prediction = base_prediction + np.random.uniform(-10, 10)
    
    lower_bound = max(0, int(base_prediction * 0.75)) # Wider range for a seasonality-only model
    upper_bound = int(base_prediction * 1.25)
    return lower_bound, upper_bound


# --- Page 1: Data Statistics and Analysis (No changes needed) ---

def page_statistics(df, df_weekly):
    """Displays comprehensive statistics and visualizations of the dengue data."""
    st.title('Dengue Data Analysis & Statistics 📊')
    st.markdown('Explore the historical trends of dengue cases and deaths from 2016 to 2021.')
    st.markdown('***')
    
    if df.empty or df_weekly.empty:
        return

    # 1. Key Metrics (Total Cases, Avg Cases, Mortality)
    total_cases = df['cases'].sum()
    total_deaths = df['deaths'].sum()
    mortality_rate = (total_deaths / total_cases) * 100 if total_cases > 0 else 0
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Recorded Cases", f"{total_cases:,.0f}", delta=f"{df['cases'].mean():.0f} avg/record")
    with metric_col2:
        st.metric("Total Recorded Deaths", f"{total_deaths:,.0f}")
    with metric_col3:
        st.metric("Case Fatality Rate", f"{mortality_rate:.2f}%")
    with metric_col4:
        st.metric("Time Span", "2016 - 2021")

    st.markdown('***')
    
    # 2. Weekly Trend Chart (Total Cases Over Time)
    st.subheader("1. National Weekly Case Trend")
    st.markdown("Visualizing the total number of cases recorded each week across the recorded years.")
    
    # Use native Streamlit chart (replaces Plotly area chart)
    st.area_chart(
        df_weekly.set_index('date')['cases'], 
        use_container_width=True,
        color="#60a5fa"
    )


    # 3. Regional Comparison (Bar Chart)
    st.subheader("2. Regional Case Distribution")
    st.markdown("Comparing the cumulative number of dengue cases across different regions.")
    
    df_region = df.groupby('Region')['cases'].sum().reset_index().sort_values(by='cases', ascending=False)
    df_region = df_region.set_index('Region')
    
    # Use native Streamlit chart (replaces Plotly bar chart)
    st.bar_chart(
        df_region['cases'], 
        use_container_width=True
    )

    # 4. Seasonal Analysis (Cases by Month)
    st.subheader("3. Monthly Seasonality Analysis")
    st.markdown("Analyzing how the average weekly case count changes throughout the year.")
    
    # Group by month name and calculate mean weekly cases
    # We use the original DataFrame for better seasonality granularity
    df_monthly = df.groupby('month_name', sort=False)['cases'].mean().reset_index()
    
    # Define correct month order for sorting the plot
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df_monthly['month_name'] = pd.Categorical(df_monthly['month_name'], categories=month_order, ordered=True)
    df_monthly = df_monthly.sort_values('month_name')
    df_monthly = df_monthly.set_index('month_name')

    # Use native Streamlit chart (replaces Plotly line chart)
    st.line_chart(
        df_monthly['cases'],
        use_container_width=True,
        color="#ef4444"
    )
    st.info("💡 **Insight:** The peak months for dengue cases appear to be during the rainy season, typically from July to October. This seasonality is a crucial feature for the prediction model.")


# --- Page 2: Risk Prediction Tool (UPDATED) ---

def page_prediction():
    """Displays the user input forms and the prediction results."""
    st.title('Dengue Risk Predictor Tool 🔬')
    st.markdown('Input the time context (Month and Week) to get a high-risk probability assessment and predicted case range for the coming week.')
    st.markdown('***')
    
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.container():
            st.subheader('Prediction Parameters')
            st.markdown('Input the time context for the week you wish to predict.')
            
            # --- Temporal Input Fields (Month & Week of Month) ---
            months = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 
                'May': 5, 'June': 6, 'July': 7, 'August': 8, 
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            
            # 1. Month input
            selected_month_name = st.selectbox(
                'Month of Prediction:', 
                options=list(months.keys()), 
                index=6, # Default to July
                key='month_select'
            )
            c_month = months[selected_month_name]

            # 2. Week of Month input (1-5) - user friendly
            c_week_of_month = st.number_input(
                'Week of Month (1-5):', 
                min_value=1, 
                max_value=5, 
                value=3, 
                step=1, 
                key='week_of_month'
            )
            
            # FEATURE ENGINEERING: Convert user-friendly input into a model-compatible feature
            c_simulated_week_of_year = (c_month - 1) * 4 + c_week_of_month
            st.caption(f"**Internal Feature:** Simulated Week of Year = {c_simulated_week_of_year}")
            
            st.markdown('---')
            
            # --- Case Input Fields (REMOVED) ---
            
            # Prediction Button
            if st.button('Predict Risk & Cases', use_container_width=True):
                
                # --- Case Range Prediction (Simulation) ---
                # Now only takes the temporal feature
                lower_cases, upper_cases = predict_case_range(c_simulated_week_of_year)
                st.session_state.lower_cases = lower_cases
                st.session_state.upper_cases = upper_cases
                
                if loaded_model:
                    # --- Probability Prediction Logic ---
                    # Must include all features from the original model training, even with dummy values (0), 
                    # as the MockModel's logic is now updated to ignore them.
                    input_data = pd.DataFrame({
                        'month': [c_month], 
                        'simulated_week_of_year': [c_simulated_week_of_year], 
                        'cases_last_week': [0], # Dummy value for model compatibility
                        'cases_2_weeks_ago': [0] # Dummy value for model compatibility
                    })

                    prob = loaded_model.predict_proba(input_data)[0][1] * 100

                    # Store result in session state
                    st.session_state.prediction_made = True
                    st.session_state.probability = prob
                    st.session_state.month = selected_month_name
                    st.session_state.simulated_week = c_simulated_week_of_year
                    # Removed: st.session_state.last_week_cases and st.session_state.two_weeks_cases
                else:
                    st.session_state.prediction_made = True 
                    st.session_state.probability = 0 
                    st.error("Cannot predict risk probability: Model file not available.")


    with col2:
        
        # --- Prediction Output Display ---
        if 'prediction_made' in st.session_state and st.session_state.prediction_made:
            
            prob = st.session_state.probability
            risk_decision, risk_icon, color_class = get_risk_style(prob)
            
            # 1. Binary Classification Decision
            st.subheader(f'{risk_icon} Prediction Result for {st.session_state.month}')
            st.markdown(f"""
            <div class="decision-card {color_class}">
                {risk_icon} {risk_decision}
            </div>
            """, unsafe_allow_html=True)

            # 2. Probability Card
            st.markdown(f"""
            <div class="probability-card" style="background-color: #334155;">
                Probability of High-Chance Dengue Period: 
                <span style="font-size: 1.5em; font-weight: 700; color: #60a5fa;">{prob:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Risk Meter
            st.markdown('<h3 style="margin-top: 30px; color: #93c5fd;">Risk Meter</h3>', unsafe_allow_html=True)
            bar_color = '#10b981' if color_class == 'low-risk-color' else ('#fbbf24' if color_class == 'moderate-risk-color' else '#ef4444')
            bar_text_color = 'black' if color_class == 'moderate-risk-color' else 'white'

            st.markdown(f"""
            <div style="
                width: 100%;
                height: 35px;
                background-color: #334155;
                border-radius: 18px;
                overflow: hidden;
                margin-top: 15px;
                margin-bottom: 25px;
                box-shadow: inset 0 0 8px rgba(0,0,0,0.4);
            ">
                <div style="
                    width: {prob}%;
                    height: 100%;
                    background-color: {bar_color};
                    border-radius: 18px;
                    transition: width 0.8s ease-in-out;
                    text-align: center;
                    line-height: 35px;
                    color: {bar_text_color};
                    font-weight: 700;
                    min-width: 30px; 
                ">
                    {prob:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4. Case Range Prediction (SIMULATION)
            st.markdown('<h3 style="margin-top: 30px;">Predicted Case Range (Simulated)</h3>')
            
            lower_cases = st.session_state.lower_cases
            upper_cases = st.session_state.upper_cases
            
            st.markdown(f"""
            <div class="probability-card" style="background-color: #334155; color: #e2e8f0; border: 1px dashed #60a5fa;">
                Estimated Case Count for the next week: 
                <br>
                <span style="font-size: 3em; color: #60a5fa; font-weight: 800;">
                {lower_cases:,.0f} - {upper_cases:,.0f}
                </span>
                <br>
                <span style="font-size: 0.8em; color: #94a3b8;">
                    (This is a **simulated forecast range** based on seasonal trends.)
                </span>
            </div>
            """, unsafe_allow_html=True)

            # 5. Contextual Info (CLEANED UP)
            current_week_of_month = st.session_state['week_of_month']
            st.markdown('---')
            st.markdown(f"""
            <p style="color: #94a3b8; font-size: 0.9em;">
                <strong>Contextual Inputs Used:</strong><br>
                Time Period: {st.session_state.month}, Week {current_week_of_month} of Month
            </p>
            """, unsafe_allow_html=True) # Previous case inputs are now completely removed from the display.

        else:
            st.markdown("""
                <div style="padding: 40px; text-align: center; border: 2px dashed #60a5fa; border-radius: 12px; margin-top: 50px;">
                    <h3 style="color: #93c5fd; border-left: none;">Awaiting Prediction...</h3>
                    <p style="color: #cbd5e1;">
                        Enter the required **Month** and **Week** in the left panel and click the 
                        'Predict Risk & Cases' button to initiate the risk assessment 
                        and see the results here.
                    </p>
                </div>
            """, unsafe_allow_html=True)


# --- Main App Logic (Page Selector) ---

def main():
    """Controls the page routing based on sidebar selection."""
    
    # Load data once at the start
    df_raw, df_weekly = load_data()

    with st.sidebar:
        st.header("Navigation")
        # Radio button for page selection
        page = st.radio(
            "Go to:",
            ('Risk Predictor', 'Data Statistics'),
            key='app_page_selector'
        )
        st.markdown('***')
        
        st.subheader("Model Info")
        st.success('Simulated Probability Model Initialized.')
        st.info("""
            **Updated Model Logic:** This tool now estimates the risk and case range 
            based **only** on the selected **Month** and **Week of Month**, relying 
            on the historical seasonality of dengue cases. 
            The model is a *mock* for demonstration.
        """)
        
        st.markdown('***')
        st.markdown("")


    if page == 'Risk Predictor':
        page_prediction()
    elif page == 'Data Statistics':
        page_statistics(df_raw, df_weekly)

# Run the main function
if __name__ == '__main__':
    main()