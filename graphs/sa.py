from graphviz import Digraph

dot = Digraph(comment='Mood Drift Detection Framework')

# ----------------------------
# INPUT LAYER
# ----------------------------
dot.node('A', 'Multimodal Input\n\n• Sleep Data\n• Behavioral Data\n• Emotional Signals')

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
dot.node('B', 'Feature Extraction\n\n• Temporal Features\n• Variance\n• Entropy\n• Cyclic Patterns')

# ----------------------------
# MODELING
# ----------------------------
dot.node('C', 'Mood Drift Modeling\n\n• Baseline Comparison\n• Deviation Detection\n• Trend Analysis')

# ----------------------------
# MVI
# ----------------------------
dot.node('D', 'Mood Variability Index (MVI)\n\n• Stability Score\n• Fluctuation Rate\n• Risk Indicator')

# ----------------------------
# EXPLAINABLE AI
# ----------------------------
dot.node('E', 'Explainable AI Layer\n\n• Feature Importance\n• Temporal Attention\n• Interpretability')

# ----------------------------
# DECISION LAYER
# ----------------------------
dot.node('F', 'Risk Classification\n\n• Low\n• Medium\n• High')

# ----------------------------
# INTERVENTION
# ----------------------------
dot.node('G', 'Early Intervention Engine\n\n• Alerts\n• Recommendations\n• Monitoring')

# ----------------------------
# FLOW CONNECTIONS
# ----------------------------
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')

# Render
dot.render('system_architecture', format='png', view=True)