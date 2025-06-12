"""
System Monitoring Dashboard
Real-time monitoring and logging interface
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from components.monitoring import get_monitor, SystemMetrics, ApplicationMetrics


def show():
    """Display the monitoring dashboard"""
    
    st.header("📊 System Monitoring Dashboard")
    st.markdown("Real-time application performance and system health monitoring")
    
    # Get monitor instance
    monitor = get_monitor()
    
    # Health status at the top
    display_health_status(monitor)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖥️ System Metrics",
        "🚀 Application Metrics",
        "📋 Logs & Events",
        "🚨 Alerts & Rules",
        "📊 Performance Analytics"
    ])
    
    with tab1:
        display_system_metrics(monitor)
    
    with tab2:
        display_application_metrics(monitor)
    
    with tab3:
        display_logs_and_events(monitor)
    
    with tab4:
        display_alerts_and_rules(monitor)
    
    with tab5:
        display_performance_analytics(monitor)


def display_health_status(monitor):
    """Display overall system health status"""
    
    health = monitor.get_health_status()
    
    # Health status banner
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = {
            'healthy': 'success',
            'warning': 'warning', 
            'critical': 'error',
            'unknown': 'info'
        }.get(health['status'], 'info')
        
        st.metric(
            "System Health",
            health['status'].title(),
            delta=f"Score: {health['score']}"
        )
    
    with col2:
        st.metric(
            "Active Alerts",
            health.get('active_alerts', 0),
            delta="🚨" if health.get('active_alerts', 0) > 0 else "✅"
        )
    
    with col3:
        latest_system = monitor.system_metrics[-1] if monitor.system_metrics else None
        cpu_value = latest_system.cpu_percent if latest_system else 0
        st.metric(
            "CPU Usage", 
            f"{cpu_value:.1f}%",
            delta="⚠️" if cpu_value > 80 else "✅"
        )
    
    with col4:
        memory_value = latest_system.memory_percent if latest_system else 0
        st.metric(
            "Memory Usage",
            f"{memory_value:.1f}%", 
            delta="⚠️" if memory_value > 85 else "✅"
        )
    
    # Issues summary
    if health.get('issues'):
        with st.expander("⚠️ Current Issues", expanded=True):
            for issue in health['issues']:
                st.warning(f"• {issue}")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh dashboard", value=True)
        if auto_refresh:
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("📤 Export Data", use_container_width=True):
            export_monitoring_data(monitor)


def display_system_metrics(monitor):
    """Display system metrics dashboard"""
    
    st.subheader("🖥️ System Performance Metrics")
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours"],
            index=0
        )
    
    with col2:
        if st.button("📊 Generate Report"):
            generate_system_report(monitor, time_range)
    
    # Get historical data
    hours_map = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24}
    hours = hours_map[time_range]
    metrics_history = monitor.get_system_metrics_history(hours)
    
    if not metrics_history:
        st.warning("No system metrics available. Please wait for data collection.")
        return
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame([m.to_dict() for m in metrics_history])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # CPU and Memory charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### CPU Usage Over Time")
        fig_cpu = go.Figure()
        fig_cpu.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cpu_percent'],
            mode='lines+markers',
            name='CPU %',
            line=dict(color='#ff6b6b', width=2),
            fill='tonexty'
        ))
        fig_cpu.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_cpu.update_layout(
            yaxis_title="CPU Usage (%)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        st.markdown("#### Memory Usage Over Time")
        fig_mem = go.Figure()
        fig_mem.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['memory_percent'],
            mode='lines+markers',
            name='Memory %',
            line=dict(color='#4ecdc4', width=2),
            fill='tonexty'
        ))
        fig_mem.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_mem.update_layout(
            yaxis_title="Memory Usage (%)",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig_mem, use_container_width=True)
    
    # Disk and Network charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Disk Usage")
        latest_metrics = metrics_history[-1]
        fig_disk = go.Figure(data=[
            go.Pie(
                labels=['Used', 'Free'],
                values=[
                    latest_metrics.disk_used,
                    latest_metrics.disk_total - latest_metrics.disk_used
                ],
                hole=0.3,
                marker_colors=['#ff7675', '#74b9ff']
            )
        ])
        fig_disk.update_layout(
            title=f"Disk Usage: {latest_metrics.disk_percent:.1f}% ({latest_metrics.disk_used:.1f}GB / {latest_metrics.disk_total:.1f}GB)",
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig_disk, use_container_width=True)
    
    with col2:
        st.markdown("#### Network I/O Over Time")
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['network_recv'] / 1024 / 1024,  # Convert to MB
            mode='lines',
            name='Received (MB)',
            line=dict(color='#55a3ff', width=2)
        ))
        fig_net.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['network_sent'] / 1024 / 1024,  # Convert to MB
            mode='lines',
            name='Sent (MB)',
            line=dict(color='#fd79a8', width=2)
        ))
        fig_net.update_layout(
            yaxis_title="Network I/O (MB)",
            height=300,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_net, use_container_width=True)
    
    # System details table
    with st.expander("📋 Current System Details"):
        latest = metrics_history[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**System Resources**")
            st.write(f"• CPU Usage: {latest.cpu_percent:.1f}%")
            st.write(f"• Memory: {latest.memory_used:.0f}MB / {latest.memory_total:.0f}MB")
            st.write(f"• Memory Usage: {latest.memory_percent:.1f}%")
        
        with col2:
            st.markdown("**Storage**")
            st.write(f"• Disk Usage: {latest.disk_percent:.1f}%")
            st.write(f"• Used Space: {latest.disk_used:.1f}GB")
            st.write(f"• Total Space: {latest.disk_total:.1f}GB")
        
        with col3:
            st.markdown("**Network**")
            st.write(f"• Data Sent: {latest.network_sent / 1024 / 1024:.1f}MB")
            st.write(f"• Data Received: {latest.network_recv / 1024 / 1024:.1f}MB")
            st.write(f"• Active Connections: {latest.active_connections}")


def display_application_metrics(monitor):
    """Display application metrics dashboard"""
    
    st.subheader("🚀 Application Performance Metrics")
    
    # Get application metrics
    app_metrics = monitor.get_application_metrics_history(1)  # Last hour
    
    if not app_metrics:
        st.warning("No application metrics available. Please wait for data collection.")
        return
    
    latest_app = app_metrics[-1]
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Crawls",
            latest_app.active_crawl_jobs,
            delta=f"{latest_app.active_crawl_jobs} running"
        )
    
    with col2:
        st.metric(
            "Total Documents",
            f"{latest_app.total_documents:,}",
            delta=f"{latest_app.total_embeddings:,} embeddings"
        )
    
    with col3:
        st.metric(
            "Search Requests/min",
            latest_app.search_requests_per_minute,
            delta="requests"
        )
    
    with col4:
        st.metric(
            "Cache Hit Rate",
            f"{latest_app.cache_hit_rate:.1%}",
            delta="efficiency"
        )
    
    # Performance charts
    if len(app_metrics) > 1:
        df_app = pd.DataFrame([m.to_dict() for m in app_metrics])
        df_app['timestamp'] = pd.to_datetime(df_app['timestamp'])
        df_app = df_app.sort_values('timestamp')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### API Response Times")
            fig_api = go.Figure()
            fig_api.add_trace(go.Scatter(
                x=df_app['timestamp'],
                y=df_app['api_response_time'],
                mode='lines+markers',
                name='API Response Time',
                line=dict(color='#6c5ce7', width=2)
            ))
            fig_api.add_hline(y=3.0, line_dash="dash", line_color="orange", annotation_text="Slow Response")
            fig_api.update_layout(
                yaxis_title="Response Time (seconds)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_api, use_container_width=True)
        
        with col2:
            st.markdown("#### Database Query Times")
            fig_db = go.Figure()
            fig_db.add_trace(go.Scatter(
                x=df_app['timestamp'],
                y=df_app['database_query_time'],
                mode='lines+markers',
                name='DB Query Time',
                line=dict(color='#00b894', width=2)
            ))
            fig_db.add_hline(y=2.0, line_dash="dash", line_color="orange", annotation_text="Slow Query")
            fig_db.update_layout(
                yaxis_title="Query Time (seconds)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_db, use_container_width=True)
    
    # Error and warning trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Error Rate")
        if len(app_metrics) > 1:
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Bar(
                x=df_app['timestamp'],
                y=df_app['error_count'],
                name='Errors',
                marker_color='#e17055'
            ))
            fig_errors.update_layout(
                yaxis_title="Error Count (5 min window)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_errors, use_container_width=True)
        else:
            st.metric("Current Errors", latest_app.error_count)
    
    with col2:
        st.markdown("#### Warning Rate")
        if len(app_metrics) > 1:
            fig_warnings = go.Figure()
            fig_warnings.add_trace(go.Bar(
                x=df_app['timestamp'],
                y=df_app['warning_count'],
                name='Warnings',
                marker_color='#fdcb6e'
            ))
            fig_warnings.update_layout(
                yaxis_title="Warning Count (5 min window)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_warnings, use_container_width=True)
        else:
            st.metric("Current Warnings", latest_app.warning_count)


def display_logs_and_events(monitor):
    """Display logs and events dashboard"""
    
    st.subheader("📋 System Logs & Events")
    
    # Log filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_filter = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours"],
            index=0
        )
    
    with col2:
        level_filter = st.selectbox(
            "Log Level",
            ["All Levels", "ERROR", "WARNING", "INFO", "DEBUG"],
            index=0
        )
    
    with col3:
        max_entries = st.number_input(
            "Max Entries",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
    
    with col4:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    
    # Get log entries
    hours_map = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24}
    hours = hours_map[time_filter]
    
    level = None if level_filter == "All Levels" else level_filter
    log_entries = monitor.get_log_entries(hours, level)[:max_entries]
    
    if not log_entries:
        st.info("No log entries found for the selected criteria.")
        return
    
    # Log level summary
    level_counts = {}
    for entry in log_entries:
        level = entry.get('level', 'UNKNOWN')
        level_counts[level] = level_counts.get(level, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ERROR", level_counts.get('ERROR', 0))
    with col2:
        st.metric("WARNING", level_counts.get('WARNING', 0))
    with col3:
        st.metric("INFO", level_counts.get('INFO', 0))
    with col4:
        st.metric("DEBUG", level_counts.get('DEBUG', 0))
    
    # Log entries table
    st.markdown("#### Recent Log Entries")
    
    # Create DataFrame for better display
    if log_entries:
        df_logs = pd.DataFrame(log_entries)
        df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
        df_logs = df_logs.sort_values('timestamp', ascending=False)
        
        # Display logs with color coding
        for _, log in df_logs.iterrows():
            level = log.get('level', 'INFO')
            timestamp = log.get('timestamp', datetime.now()).strftime('%H:%M:%S')
            logger = log.get('logger', 'unknown')
            message = log.get('message', '')
            
            # Color code by level
            if level == 'ERROR':
                st.error(f"**{timestamp}** [{level}] {logger}: {message}")
            elif level == 'WARNING':
                st.warning(f"**{timestamp}** [{level}] {logger}: {message}")
            elif level == 'INFO':
                st.info(f"**{timestamp}** [{level}] {logger}: {message}")
            else:
                st.text(f"{timestamp} [{level}] {logger}: {message}")
    
    # Export logs option
    with st.expander("📤 Export Logs"):
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Format", ["JSON", "CSV", "TXT"])
        
        with col2:
            if st.button("Download Logs"):
                export_logs(log_entries, export_format)


def display_alerts_and_rules(monitor):
    """Display alerts and alert rules management"""
    
    st.subheader("🚨 Alerts & Monitoring Rules")
    
    # Active alerts section
    if monitor.active_alerts:
        st.markdown("#### 🚨 Active Alerts")
        
        for alert_name, start_time in monitor.active_alerts.items():
            duration = datetime.now() - start_time
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.error(f"**{alert_name}**")
                    st.write(f"Active for: {str(duration).split('.')[0]}")
                
                with col2:
                    st.write(f"Since: {start_time.strftime('%H:%M:%S')}")
                
                with col3:
                    if st.button(f"Acknowledge", key=f"ack_{alert_name}"):
                        # Acknowledge alert (could add to database)
                        st.success("Alert acknowledged")
        
        st.markdown("---")
    else:
        st.success("✅ No active alerts")
    
    # Alert rules management
    st.markdown("#### ⚙️ Alert Rules Configuration")
    
    # Display current rules
    rules_df = []
    for rule in monitor.alert_rules:
        rules_df.append({
            'Name': rule.name,
            'Metric': rule.metric,
            'Condition': f"{rule.operator} {rule.threshold}",
            'Duration': f"{rule.duration_minutes} min",
            'Enabled': rule.enabled
        })
    
    if rules_df:
        st.dataframe(pd.DataFrame(rules_df), use_container_width=True)
    
    # Add new rule
    with st.expander("➕ Add New Alert Rule"):
        col1, col2 = st.columns(2)
        
        with col1:
            rule_name = st.text_input("Rule Name")
            metric = st.selectbox(
                "Metric",
                ["cpu_percent", "memory_percent", "disk_percent", "error_count", 
                 "api_response_time", "database_query_time"]
            )
            operator = st.selectbox("Operator", [">", "<", ">=", "<=", "=="])
        
        with col2:
            threshold = st.number_input("Threshold", value=80.0)
            duration = st.number_input("Duration (minutes)", value=5, min_value=1)
            enabled = st.checkbox("Enabled", value=True)
        
        if st.button("Add Rule") and rule_name:
            from components.monitoring import AlertRule
            new_rule = AlertRule(rule_name, metric, operator, threshold, duration, enabled)
            monitor.alert_rules.append(new_rule)
            st.success(f"Added alert rule: {rule_name}")
            st.rerun()


def display_performance_analytics(monitor):
    """Display performance analytics and insights"""
    
    st.subheader("📊 Performance Analytics")
    
    # Get extended metrics history
    system_history = monitor.get_system_metrics_history(24)  # Last 24 hours
    app_history = monitor.get_application_metrics_history(24)
    
    if not system_history or not app_history:
        st.warning("Insufficient data for analytics. Please wait for more data collection.")
        return
    
    # Performance insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Performance Insights")
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in system_history) / len(system_history)
        avg_memory = sum(m.memory_percent for m in system_history) / len(system_history)
        avg_api_time = sum(m.api_response_time for m in app_history) / len(app_history)
        
        insights = []
        
        if avg_cpu > 70:
            insights.append("🔥 High average CPU usage detected")
        if avg_memory > 80:
            insights.append("💾 High memory consumption pattern")
        if avg_api_time > 2.0:
            insights.append("🐌 API responses slower than optimal")
        
        # Error patterns
        total_errors = sum(m.error_count for m in app_history)
        if total_errors > 50:
            insights.append("❌ High error rate detected")
        
        if insights:
            for insight in insights:
                st.warning(insight)
        else:
            st.success("✅ Performance looks healthy!")
    
    with col2:
        st.markdown("#### 📈 Key Performance Indicators")
        
        # KPI metrics
        uptime_hours = len(system_history) * 0.5  # Assuming 30-second intervals
        total_searches = sum(m.search_requests_per_minute for m in app_history)
        avg_cache_hit = sum(m.cache_hit_rate for m in app_history) / len(app_history)
        
        st.metric("System Uptime", f"{uptime_hours:.1f} hours")
        st.metric("Total Searches", f"{total_searches:,}")
        st.metric("Avg Cache Hit Rate", f"{avg_cache_hit:.1%}")
        st.metric("Data Points Collected", f"{len(system_history):,}")
    
    # Correlation analysis
    if len(system_history) > 10 and len(app_history) > 10:
        st.markdown("#### 🔗 Performance Correlations")
        
        # Create correlation matrix
        df_system = pd.DataFrame([m.to_dict() for m in system_history])
        df_app = pd.DataFrame([m.to_dict() for m in app_history])
        
        # Merge on timestamp
        df_system['timestamp'] = pd.to_datetime(df_system['timestamp'])
        df_app['timestamp'] = pd.to_datetime(df_app['timestamp'])
        
        df_merged = pd.merge_asof(
            df_system.sort_values('timestamp'),
            df_app.sort_values('timestamp'),
            on='timestamp',
            suffixes=('_sys', '_app')
        )
        
        # Calculate correlations
        correlation_pairs = [
            ('cpu_percent', 'api_response_time'),
            ('memory_percent', 'database_query_time'),
            ('cpu_percent', 'error_count'),
        ]
        
        for metric1, metric2 in correlation_pairs:
            if metric1 in df_merged.columns and metric2 in df_merged.columns:
                corr = df_merged[metric1].corr(df_merged[metric2])
                
                if abs(corr) > 0.3:  # Show only significant correlations
                    correlation_text = f"{metric1} ↔ {metric2}: {corr:.2f}"
                    if corr > 0.5:
                        st.warning(f"🔴 Strong positive correlation: {correlation_text}")
                    elif corr < -0.5:
                        st.info(f"🔵 Strong negative correlation: {correlation_text}")
                    else:
                        st.text(f"🟡 Moderate correlation: {correlation_text}")


def export_monitoring_data(monitor):
    """Export monitoring data"""
    
    st.markdown("### 📤 Export Monitoring Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours = st.number_input("Hours of data", min_value=1, max_value=168, value=24)
    
    with col2:
        export_format = st.selectbox("Format", ["JSON", "CSV"])
    
    if st.button("Generate Export"):
        try:
            export_data = monitor.export_metrics(hours)
            
            if export_format == "JSON":
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                # Convert to CSV format
                system_df = pd.DataFrame(export_data.get('system_metrics', []))
                app_df = pd.DataFrame(export_data.get('application_metrics', []))
                
                csv_data = f"System Metrics:\n{system_df.to_csv(index=False)}\n\nApplication Metrics:\n{app_df.to_csv(index=False)}"
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            st.success("Export ready for download!")
            
        except Exception as e:
            st.error(f"Export failed: {e}")


def export_logs(log_entries: List[Dict], format: str):
    """Export log entries"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == "JSON":
        json_str = json.dumps(log_entries, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON Logs",
            data=json_str,
            file_name=f"logs_{timestamp}.json",
            mime="application/json"
        )
    elif format == "CSV":
        df = pd.DataFrame(log_entries)
        csv_str = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Logs",
            data=csv_str,
            file_name=f"logs_{timestamp}.csv",
            mime="text/csv"
        )
    else:  # TXT
        txt_lines = []
        for entry in log_entries:
            timestamp_str = entry.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            level = entry.get('level', 'INFO')
            logger = entry.get('logger', 'unknown')
            message = entry.get('message', '')
            txt_lines.append(f"{timestamp_str} [{level}] {logger}: {message}")
        
        txt_str = '\n'.join(txt_lines)
        st.download_button(
            label="📥 Download TXT Logs",
            data=txt_str,
            file_name=f"logs_{timestamp}.txt",
            mime="text/plain"
        )


def generate_system_report(monitor, time_range: str):
    """Generate a comprehensive system report"""
    
    st.markdown("### 📊 System Performance Report")
    
    hours_map = {"Last Hour": 1, "Last 6 Hours": 6, "Last 24 Hours": 24}
    hours = hours_map[time_range]
    
    system_history = monitor.get_system_metrics_history(hours)
    app_history = monitor.get_application_metrics_history(hours)
    
    if not system_history:
        st.error("No data available for report generation")
        return
    
    # Generate report summary
    report_data = {
        'report_period': time_range,
        'report_generated': datetime.now().isoformat(),
        'data_points': len(system_history),
        'summary': {
            'avg_cpu': sum(m.cpu_percent for m in system_history) / len(system_history),
            'max_cpu': max(m.cpu_percent for m in system_history),
            'avg_memory': sum(m.memory_percent for m in system_history) / len(system_history),
            'max_memory': max(m.memory_percent for m in system_history),
            'disk_usage': system_history[-1].disk_percent if system_history else 0,
        }
    }
    
    # Display report
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### System Summary")
        st.write(f"**Report Period:** {time_range}")
        st.write(f"**Data Points:** {report_data['data_points']}")
        st.write(f"**Average CPU:** {report_data['summary']['avg_cpu']:.1f}%")
        st.write(f"**Peak CPU:** {report_data['summary']['max_cpu']:.1f}%")
    
    with col2:
        st.markdown("#### Performance Metrics")
        st.write(f"**Average Memory:** {report_data['summary']['avg_memory']:.1f}%")
        st.write(f"**Peak Memory:** {report_data['summary']['max_memory']:.1f}%")
        st.write(f"**Current Disk Usage:** {report_data['summary']['disk_usage']:.1f}%")
    
    # Download report
    json_report = json.dumps(report_data, indent=2, default=str)
    st.download_button(
        label="📥 Download Report",
        data=json_report,
        file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


if __name__ == "__main__":
    show()