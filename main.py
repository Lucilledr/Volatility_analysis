import tkinter as tk
from tkinter import ttk
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analyzer import StockAnalyzer
from product_announcements import ProductAnnouncements
from tkinter import ttk, scrolledtext
import numpy as np
from sklearn.linear_model import LinearRegression
import os


class StockVolatilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tech Stock Volatility Analyzer")
        self.root.geometry("1200x900")

        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Modern looking theme

        # Configure custom styles
        self.style.configure("Title.TLabel", font=("Helvetica", 24, "bold"))
        self.style.configure("Subtitle.TLabel", font=("Helvetica", 12))
        self.style.configure(
            "Custom.TButton",
            font=("Helvetica", 11),
            padding=10,
            background="#2196F3",
            foreground="white",
        )
        self.style.configure("Custom.TRadiobutton", font=("Helvetica", 11))

        # Initialize components
        self.setup_ui()
        self.analyzer = StockAnalyzer()
        self.announcements = ProductAnnouncements()

        self.company_data = {}  # Store data for each company

    def setup_ui(self):
        # Main title
        title_frame = ttk.Frame(self.root, padding="20 20 20 0")
        title_frame.pack(fill="x")

        ttk.Label(
            title_frame, text="Tech Stock Volatility Analyzer", style="Title.TLabel"
        ).pack()

        ttk.Label(
            title_frame,
            text="Analyze the impact of product announcements on stock volatility",
            style="Subtitle.TLabel",
        ).pack(pady=5)

        # Create main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill="both", expand=True)

        # Company selection with better styling
        companies_frame = ttk.LabelFrame(
            main_container, text="Select Company", padding=15
        )
        companies_frame.pack(fill="x", padx=10, pady=10)

        # Add a sub-frame for the "All" button
        select_all_frame = ttk.Frame(companies_frame)
        select_all_frame.pack(fill="x", padx=5, pady=(0, 5))

        # Add "All" button
        self.all_var = tk.BooleanVar(value=False)
        self.all_button = ttk.Checkbutton(
            select_all_frame,
            text="All Companies",
            variable=self.all_var,
            command=self.toggle_all_companies,
            style="Custom.TCheckbutton",
        )
        self.all_button.pack(side="left", padx=20)

        # Add separator
        ttk.Separator(companies_frame, orient="horizontal").pack(
            fill="x", padx=5, pady=5
        )

        # Companies checkboxes frame
        companies_checks_frame = ttk.Frame(companies_frame)
        companies_checks_frame.pack(fill="x")

        # Replace radio buttons with checkboxes
        self.company_vars = {}
        companies = [
            ("Apple", "AAPL"),
            ("Microsoft", "MSFT"),
            ("Google", "GOOGL"),
            ("Amazon", "AMZN"),
            ("Tesla", "TSLA"),
            ("Meta", "META"),
            ("Nvidia", "NVDA"),
            ("Tencent", "TCEHY"),
            ("Broadcom", "AVGO"),
        ]

        for name, symbol in companies:
            var = tk.BooleanVar()
            self.company_vars[symbol] = var
            check = ttk.Checkbutton(
                companies_checks_frame,
                text=name,
                variable=var,
                command=self.check_all_status,
                style="Custom.TCheckbutton",
            )
            check.pack(side="left", padx=20, pady=5)

        # Date range selection with better organization
        dates_frame = ttk.LabelFrame(
            main_container, text="Select Date Range", padding=15
        )
        dates_frame.pack(fill="x", padx=10, pady=10)

        # Create grid layout for dates
        date_grid = ttk.Frame(dates_frame)
        date_grid.pack(pady=5)

        ttk.Label(date_grid, text="Start Date:").grid(row=0, column=0, padx=5)
        self.start_date = ttk.Entry(date_grid, width=12)
        self.start_date.grid(row=0, column=1, padx=5)

        ttk.Label(date_grid, text="End Date:").grid(row=0, column=2, padx=5)
        self.end_date = ttk.Entry(date_grid, width=12)
        self.end_date.grid(row=0, column=3, padx=5)

        # Set default dates
        self.start_date.insert(
            0, (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        )
        self.end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))

        # Add Analysis Options frame
        analysis_frame = ttk.LabelFrame(
            main_container, text="Analysis Options", padding=15
        )
        analysis_frame.pack(fill="x", padx=10, pady=10)

        # Add buttons
        self.analyze_button = ttk.Button(
            analysis_frame,
            text="Analyze Volatility",
            style="Custom.TButton",
            command=self.analyze_impact
        )
        self.analyze_button.pack(side="left", padx=5)

        self.regression_button = ttk.Button(
            analysis_frame,
            text="Perform Regression Analysis",
            style="Custom.TButton",
            command=self.perform_regression_analysis
        )
        self.regression_button.pack(side="left", padx=5)

        # Results area
        self.results_frame = ttk.Frame(main_container)
        self.results_frame.pack(fill="both", expand=True)

    def analyze_impact(self):
        start = self.start_date.get()
        end = self.end_date.get()

        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Get selected companies
        selected_companies = [
            symbol for symbol, var in self.company_vars.items() if var.get()
        ]

        if not selected_companies:
            # Show error message if no company selected
            ttk.Label(
                self.results_frame,
                text="Please select at least one company",
                style="Subtitle.TLabel",
            ).pack(pady=20)
            return

        # Get data for each selected company
        self.company_data = {}
        for symbol in selected_companies:
            stock_data = self.analyzer.get_stock_data(symbol, start, end)
            announcements = self.announcements.get_announcements(symbol, start, end)
            volatility_data = self.analyzer.calculate_volatility(
                stock_data, announcements
            )
            self.company_data[symbol] = {
                "volatility_data": volatility_data,
                "announcements": announcements,
            }

        # Create visualization
        self.create_visualization()

        # Display summary
        self.display_summary()

    def create_visualization(self):
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Stock Prices", "Volatility"),
            vertical_spacing=0.2,
            row_heights=[0.5, 0.5],
        )

        # Color palette for different companies
        colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FFC107"]

        for idx, (symbol, data) in enumerate(self.company_data.items()):
            volatility_data = data["volatility_data"]
            announcements = data["announcements"]
            color = colors[idx % len(colors)]

            # Add price trace
            fig.add_trace(
                go.Scatter(
                    x=volatility_data.index,
                    y=volatility_data["Close"],
                    name=f"{symbol} Price",
                    line=dict(color=color),
                    hovertemplate="<b>Date</b>: %{x}<br>"
                    + "<b>Price</b>: $%{y:.2f}<br><extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Add volatility trace
            fig.add_trace(
                go.Scatter(
                    x=volatility_data.index,
                    y=volatility_data["Volatility"],
                    name=f"{symbol} Volatility",
                    line=dict(color=color),
                    hovertemplate="<b>Date</b>: %{x}<br>"
                    + "<b>Volatility</b>: %{y:.2f}%<br><extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Add announcement markers
            for date in announcements:
                date_str = date.strftime("%Y-%m-%d")
                if date_str in self.announcements.announcements[symbol]:
                    announcement_text = self.announcements.announcements[symbol][
                        date_str
                    ]

                    # Add vertical lines for announcements
                    fig.add_vline(
                        x=date,
                        line_dash="dash",
                        line_color=color,
                        opacity=0.3,
                        row=1,
                        col=1,
                    )
                    fig.add_vline(
                        x=date,
                        line_dash="dash",
                        line_color=color,
                        opacity=0.3,
                        row=2,
                        col=1,
                    )

                    # Add annotations for announcements
                    fig.add_annotation(
                        x=date,
                        y=volatility_data["Close"].max(),
                        text=f"{symbol}: {announcement_text}",
                        showarrow=True,
                        arrowhead=1,
                        row=1,
                        col=1,
                        font=dict(size=10),
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8,
                    )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
            hovermode="x unified",
            template="plotly_white",
            margin=dict(t=30, l=10, r=10, b=10),
        )

        # Update axes labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        # Save the plot as HTML
        html_file = "stock_analysis.html"
        fig.write_html(html_file)
        
        # Create a message frame
        message_frame = ttk.Frame(self.results_frame)
        message_frame.pack(pady=20)
        
        # Add informative message
        message = ttk.Label(
            message_frame,
            text="Analysis complete! The interactive chart has been saved.",
            style='Subtitle.TLabel'
        )
        message.pack(pady=(0, 10))
        
        # Add file location message
        file_path = os.path.abspath(html_file)
        location_msg = ttk.Label(
            message_frame,
            text=f"File location: {file_path}",
            style='Subtitle.TLabel'
        )
        location_msg.pack(pady=(0, 10))
        
        # Add buttons frame
        buttons_frame = ttk.Frame(message_frame)
        buttons_frame.pack()
        
        # Function to open in default browser
        def open_in_browser():
            import webbrowser
            try:
                webbrowser.open('file://' + file_path)
            except Exception as e:
                error_label.config(text=f"Error opening browser: {str(e)}")
        
        # Function to open in specific browsers
        def open_in_chrome():
            import subprocess
            try:
                subprocess.run(['open', '-a', 'Google Chrome', file_path])
            except Exception as e:
                error_label.config(text=f"Error opening Chrome: {str(e)}")
                
        def open_in_safari():
            import subprocess
            try:
                subprocess.run(['open', '-a', 'Safari', file_path])
            except Exception as e:
                error_label.config(text=f"Error opening Safari: {str(e)}")
        
        # Add buttons for different options
        ttk.Button(
            buttons_frame,
            text="Open in Default Browser",
            command=open_in_browser,
            style='Custom.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Open in Chrome",
            command=open_in_chrome,
            style='Custom.TButton'
        ).pack(side='left', padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Open in Safari",
            command=open_in_safari,
            style='Custom.TButton'
        ).pack(side='left', padx=5)
        
        # Add error label
        error_label = ttk.Label(
            message_frame,
            text="",
            foreground='red',
            style='Subtitle.TLabel'
        )
        error_label.pack(pady=(10, 0))
        
        # Copy path button
        def copy_path():
            self.root.clipboard_clear()
            self.root.clipboard_append(file_path)
            copy_label.config(text="Path copied to clipboard!")
            self.root.after(2000, lambda: copy_label.config(text=""))
        
        ttk.Button(
            message_frame,
            text="Copy File Path",
            command=copy_path,
            style='Custom.TButton'
        ).pack(pady=(10, 0))
        
        copy_label = ttk.Label(
            message_frame,
            text="",
            style='Subtitle.TLabel'
        )
        copy_label.pack()

    def display_summary(self):
        summary_frame = ttk.Frame(self.results_frame)
        summary_frame.pack(pady=15, padx=10)

        for symbol, data in self.company_data.items():
            volatility_data = data["volatility_data"]
            stats_text = (
                f"{symbol} Summary Statistics:\n"
                f"Average Volatility: {volatility_data['Volatility'].mean():.2f}%\n"
                f"Maximum Volatility: {volatility_data['Volatility'].max():.2f}%\n"
                f"Minimum Volatility: {volatility_data['Volatility'].min():.2f}%\n"
                f"Standard Deviation: {volatility_data['Volatility'].std():.2f}%\n\n"
            )

            summary = ttk.Label(
                summary_frame, text=stats_text, justify="left", font=("Helvetica", 11)
            )
            summary.pack(pady=5)

    def toggle_all_companies(self):
        """Toggle all company checkboxes based on the All button state"""
        state = self.all_var.get()
        for var in self.company_vars.values():
            var.set(state)

    def check_all_status(self):
        """Check if all companies are selected and update the All button accordingly"""
        all_selected = all(var.get() for var in self.company_vars.values())
        self.all_var.set(all_selected)

    def perform_regression_analysis(self):
        """Perform regression analysis for each selected company"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Create text widget for results
        results_text = scrolledtext.ScrolledText(
            self.results_frame,
            height=15,
            width=80,
            font=("Courier", 10)
        )
        results_text.pack(pady=10, padx=10)

        # Check if we have any data to analyze
        if not self.company_data:
            results_text.insert('end', "Please run 'Analyze Volatility' first before performing regression analysis.\n")
            results_text.configure(state='disabled')
            return

        # Counter for successful analyses
        analyses_performed = 0

        # Analyze each selected company
        for symbol, data in self.company_data.items():
            try:
                volatility_data = data['volatility_data']
                announcements = data['announcements']
                
                # Debug information
                results_text.insert('end', f"\nAnalyzing {symbol}...\n")
                results_text.insert('end', f"Number of announcements: {len(announcements)}\n")
                results_text.insert('end', f"Data range: {volatility_data.index.min()} to {volatility_data.index.max()}\n\n")
                
                if not announcements:
                    results_text.insert('end', f"No announcements found for {symbol} in the selected date range.\n")
                    continue
                
                # Perform regression analysis
                regression_results = self.analyzer.perform_event_impact_analysis(
                    volatility_data,
                    announcements
                )
                
                # Display results
                results_text.insert('end', f"\n{'='*80}\n")
                results_text.insert('end', f"Regression Analysis Results for {symbol}\n")
                results_text.insert('end', f"{'='*80}\n\n")
                
                if len(regression_results) == 0:
                    results_text.insert('end', "No valid events found for analysis in the selected date range.\n")
                    continue
                
                # Calculate average impact
                avg_impact = regression_results['Volatility_Change'].mean()
                avg_r2 = regression_results['R_squared'].mean()
                
                results_text.insert('end', f"Average Impact on Volatility: {avg_impact:.2f}%\n")
                results_text.insert('end', f"Average R-squared: {avg_r2:.3f}\n\n")
                
                results_text.insert('end', "Event-by-Event Analysis:\n")
                results_text.insert('end', "-" * 80 + "\n")
                
                for _, row in regression_results.iterrows():
                    date_str = row['Date']
                    event_description = self.announcements.announcements[symbol].get(date_str, "No description available")
                    
                    results_text.insert('end', f"Date: {date_str}\n")
                    results_text.insert('end', f"Event: {event_description}\n")
                    results_text.insert('end', f"Pre-Event Volatility: {row['Pre_Event_Volatility']:.2f}%\n")
                    results_text.insert('end', f"Post-Event Volatility: {row['Post_Event_Volatility']:.2f}%\n")
                    results_text.insert('end', f"Volatility Change: {row['Volatility_Change']:.2f}%\n")
                    results_text.insert('end', f"Impact Coefficient: {row['Impact_Coefficient']:.3f}\n")
                    results_text.insert('end', f"R-squared: {row['R_squared']:.3f}\n")
                    results_text.insert('end', "-" * 80 + "\n")
                
                analyses_performed += 1

            except Exception as e:
                results_text.insert('end', f"\nError analyzing {symbol}: {str(e)}\n")
                import traceback
                results_text.insert('end', traceback.format_exc())
        
        if analyses_performed == 0:
            results_text.insert('end', "\nNo analyses could be performed. Please check your data and date range.\n")
        
        # Make text read-only
        results_text.configure(state='disabled')
        
        # Ensure the text widget is visible
        results_text.see('1.0')


if __name__ == "__main__":
    root = tk.Tk()
    app = StockVolatilityApp(root)
    root.mainloop()
