# Hippocampal circuit reference card.
# Called via exec() from notebook cell-0; __vsc_ipynb_file__ is in caller scope.
from IPython.display import HTML, display
from pathlib import Path

_html = Path(__vsc_ipynb_file__).parent.parent / 'visualizations' / 'hippocampal_circuit_neutral.html'
display(HTML(_html.read_text()))
