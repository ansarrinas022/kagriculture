"""
Report Generator Module for Kaggriculture.
Generates comprehensive executive reports in styled HTML and structured CSV formats.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class ReportGenerator:
    """Generates executive HTML and CSV exportable analytical reports."""

    @staticmethod
    def generate_html_report(
        summary_metrics: Dict[str, Any],
        leaderboard_df: pd.DataFrame,
        insights: List[Dict[str, Any]],
        validation_report: Dict[str, Any],
        active_filters: Dict[str, Any],
        agent_comp_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Creates a standalone, publication-quality HTML executive report
        with embedded CSS styling, executive summary, tables, and warnings.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format Leaderboard Table
        lb_rows = ""
        if not leaderboard_df.empty:
            for _, r in leaderboard_df.head(10).iterrows():
                lb_rows += f"""
                <tr>
                    <td style="font-weight:600; color:#10B981;">{r.get('agent', 'N/A')}</td>
                    <td>{r.get('episodes', 0)}</td>
                    <td><b>${r.get('avg_score', 0):,.2f}</b></td>
                    <td>${r.get('median_score', 0):,.2f}</td>
                    <td>${r.get('max_score', 0):,.2f}</td>
                    <td><span style="color:#34D399; font-weight:600;">{r.get('win_rate', 0)}%</span></td>
                    <td>{r.get('avg_harvests', 0)}</td>
                    <td>${r.get('avg_market_sales', 0):,.2f}</td>
                </tr>
                """

        # Format Insights
        insights_html = ""
        for ins in insights:
            insights_html += f"""
            <div style="background:#132A22; border-left:4px solid #10B981; padding:16px; margin-bottom:14px; border-radius:6px;">
                <h4 style="margin:0 0 8px 0; color:#34D399; font-size:1.05rem;">{ins.get('title', 'Observation')}</h4>
                <p style="margin:0 0 6px 0; color:#ECFDF5; font-size:0.95rem;">{ins.get('observation', '')}</p>
                <div style="font-size:0.85rem; color:#9CA3AF; margin-top:8px;">
                    <span style="display:inline-block; margin-right:15px;"><b>Metric:</b> {ins.get('metric', 'N/A')}</span>
                    <span style="display:inline-block; margin-right:15px;"><b>Sample Size:</b> {ins.get('sample_size', 'N/A')}</span>
                </div>
                <div style="margin-top:8px; font-size:0.85rem; background:#0B1914; padding:8px 12px; border-radius:4px; color:#F4F6F0;">
                    <b>Interpretation:</b> {ins.get('interpretation', '')}<br>
                    <span style="color:#64748B;"><i>Limitation: {ins.get('limitation', '')}</i></span>
                </div>
            </div>
            """

        # Format Agent Comparison (if provided)
        comp_html = ""
        if agent_comp_data:
            ag_a = agent_comp_data.get("agent_a", "Agent A")
            ag_b = agent_comp_data.get("agent_b", "Agent B")
            m_stats = agent_comp_data.get("matched_stats", {})
            m_count = m_stats.get("matched_episodes_count", 0)

            if m_count > 0:
                comp_html = f"""
                <div style="background:#132A22; border:1px solid #1E3E33; border-radius:8px; padding:18px; margin-bottom:24px;">
                    <h3 style="color:#10B981; margin-top:0;">Head-to-Head Matched Comparison: {ag_a} vs {ag_b}</h3>
                    <p style="color:#9CA3AF; font-size:0.9rem;">Evaluated across <b>{m_count}</b> direct matched simulation matches.</p>
                    <table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.9rem;">
                        <thead>
                            <tr style="border-bottom:1px solid #1E3E33; text-align:left; color:#9CA3AF;">
                                <th style="padding:8px;">Metric</th>
                                <th style="padding:8px; color:#10B981;">{ag_a}</th>
                                <th style="padding:8px; color:#38BDF8;">{ag_b}</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom:1px solid #1A382E;">
                                <td style="padding:8px;">Direct Wins</td>
                                <td style="padding:8px; font-weight:bold;">{m_stats.get('wins_a')} ({m_stats.get('win_rate_a')}%)</td>
                                <td style="padding:8px; font-weight:bold;">{m_stats.get('wins_b')} ({m_stats.get('win_rate_b')}%)</td>
                            </tr>
                            <tr style="border-bottom:1px solid #1A382E;">
                                <td style="padding:8px;">Average Final Score</td>
                                <td style="padding:8px;">${m_stats.get('mean_score_a', 0):,.2f}</td>
                                <td style="padding:8px;">${m_stats.get('mean_score_b', 0):,.2f}</td>
                            </tr>
                            <tr style="border-bottom:1px solid #1A382E;">
                                <td style="padding:8px;">Median Final Score</td>
                                <td style="padding:8px;">${m_stats.get('median_score_a', 0):,.2f}</td>
                                <td style="padding:8px;">${m_stats.get('median_score_b', 0):,.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding:8px;">Score Difference (A - B)</td>
                                <td colspan="2" style="padding:8px; font-weight:bold; color:#F59E0B;">${m_stats.get('avg_score_diff', 0):+,.2f}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """

        # Format Validation Issues
        issues = validation_report.get("issues", [])
        issues_html = ""
        for iss in issues[:8]:
            color = "#EF4444" if iss["severity"] == "ERROR" else ("#F59E0B" if iss["severity"] == "WARNING" else "#38BDF8")
            issues_html += f"""
            <li style="margin-bottom:6px; color:#ECFDF5; font-size:0.88rem;">
                <span style="color:{color}; font-weight:bold;">[{iss['severity']}]</span> {iss['message']}
            </li>
            """

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Kaggriculture Executive Simulation Analysis Report</title>
    <style>
        body {{
            background-color: #0B1914;
            color: #ECFDF5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 30px;
            line-height: 1.6;
        }}
        .report-header {{
            border-bottom: 2px solid #10B981;
            padding-bottom: 16px;
            margin-bottom: 28px;
        }}
        .report-header h1 {{
            color: #10B981;
            margin: 0 0 6px 0;
            font-size: 2rem;
        }}
        .report-header p {{
            color: #9CA3AF;
            margin: 0;
            font-size: 0.95rem;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .metric-card {{
            background: #132A22;
            border: 1px solid #1E3E33;
            border-radius: 8px;
            padding: 16px;
        }}
        .metric-card .title {{
            color: #9CA3AF;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-card .val {{
            color: #10B981;
            font-size: 1.6rem;
            font-weight: bold;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            margin-bottom: 24px;
            font-size: 0.9rem;
        }}
        th {{
            background: #0E221B;
            color: #9CA3AF;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #1E3E33;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #1E3E33;
        }}
        .section-title {{
            color: #34D399;
            font-size: 1.3rem;
            margin-top: 32px;
            margin-bottom: 14px;
            border-bottom: 1px solid #1E3E33;
            padding-bottom: 6px;
        }}
        .disclaimer-box {{
            background: #0E221B;
            border: 1px solid #D97706;
            border-radius: 6px;
            padding: 14px;
            margin-top: 36px;
            font-size: 0.85rem;
            color: #FBBF24;
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1>🌱 Kaggriculture Analysis & Strategy Report</h1>
        <p>Generated on {timestamp} | Simulation Episodes Evaluation</p>
    </div>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="title">Total Episodes</div>
            <div class="val">{summary_metrics.get('total_episodes', 0):,}</div>
        </div>
        <div class="metric-card">
            <div class="title">Unique Agents</div>
            <div class="val">{summary_metrics.get('unique_agents_count', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="title">Average Score</div>
            <div class="val">${summary_metrics.get('avg_score', 0):,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="title">Highest Score</div>
            <div class="val">${summary_metrics.get('max_score', 0):,.2f}</div>
        </div>
        <div class="metric-card">
            <div class="title">Total Actions</div>
            <div class="val">{summary_metrics.get('total_actions', 0) or 'N/A'}</div>
        </div>
    </div>

    {comp_html}

    <div class="section-title">🏆 Agent Performance Leaderboard</div>
    <table>
        <thead>
            <tr>
                <th>Agent</th>
                <th>Episodes</th>
                <th>Avg Score</th>
                <th>Median</th>
                <th>Peak</th>
                <th>Win Rate</th>
                <th>Avg Harvests</th>
                <th>Avg Sales</th>
            </tr>
        </thead>
        <tbody>
            {lb_rows}
        </tbody>
    </table>

    <div class="section-title">💡 Strategic Patterns & Action Insights</div>
    {insights_html}

    <div class="section-title">🛡️ Data Quality & Audit Logs</div>
    <ul>
        {issues_html if issues_html else "<li style='color:#34D399;'>All data validation checks passed with zero issues.</li>"}
    </ul>

    <div class="disclaimer-box">
        <b>Statistical & Empirical Limitations:</b>
        All insights and performance metrics are derived empirically from the uploaded simulation records.
        Correlation does not imply causation. Simulation balance adjustments, weather seed variance, and market price dynamics
        influence outcomes. Strategy suggestions reflect historical frequency associations and do not guarantee future competitive victories.
    </div>
</body>
</html>
"""
        return html_template
