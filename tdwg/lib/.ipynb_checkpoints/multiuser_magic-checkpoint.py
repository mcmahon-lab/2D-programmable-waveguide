"""
Using a separate file since this is specific for IPython
"""

from IPython.core.magic import register_cell_magic

@register_cell_magic
def lock(line, cell):
    # Prepend and append the lock and unlock functions to the cell content
    code_to_run = "client.lock()\n"
    code_to_run += cell
    code_to_run += "\nunlock_msg=client.unlock();"

    # Execute the modified code in the global context
    get_ipython().run_cell(code_to_run)