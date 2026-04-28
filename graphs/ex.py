from graphviz import Digraph

dot = Digraph(comment='Workflow Model', format='png')

dot.attr(rankdir='LR', size='10,4')

# Nodes
dot.node('A', 'Data Collection\n\n• Physiological\n• Behavioral\n• Emotional')
dot.node('B', 'Feature Extraction\n\n• Variance\n• Entropy\n• Cyclic Patterns')
dot.node('C', 'Mood Variability Index (MVI)\n\n• Intensity\n• Frequency')
dot.node('D', 'Explainable AI\n\n• Feature Attribution\n• Attention Mapping')
dot.node('E', 'Intervention Engine\n\n• Risk Classification\n• Alerts & Recommendations')

# Connections
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')

# Render
dot.render('workflow_model', view=True)