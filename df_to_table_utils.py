import re
import pandas as pd
import numpy as np

def parse_css(html_text):
    """
    Extracts CSS from HTML and parses it into a dictionary of IDs and attributes.

    Args:
        html_text: The HTML text containing the <style> block.

    Returns:
        A dictionary where keys are CSS IDs (without '#') and 
        values are dictionaries of attributes.
    """

    # Extract the CSS text from the <style> tag using a regular expression
    match = re.search(r'<style type="text/css">(.*?)</style>', html_text, re.DOTALL)
    if not match:
        return {}  # Return an empty dictionary if no <style> tag is found

    css_text = match.group(1).strip()

    # --- The rest of the code is the same as before ---
    css_dict = {}
    for block in css_text.split('}'):
        if block.strip():
            id_matches = re.findall(r'#([\w_-]+)', block)
            if id_matches:
                attributes = {}
                for line in block.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        attributes[key.strip()] = value.strip().rstrip(';')
                for css_id in id_matches:
                    css_dict[css_id] = attributes
    return css_dict



def get_ids_to_undeline(css_dict):
    """
    Extracts the IDs that should be underlined from a CSS dictionary.

    Args:
        css_dict: A dictionary where keys are CSS IDs and values are dictionaries of attributes.

    Returns:
        A list of IDs that should be underlined.
    """
    ids_to_be_underlined = []
    for css_id, attributes in css_dict.items():
        # Check if there is a text-decoration: underline; attribute
        if 'text-decoration' in attributes:
            if 'underline' in attributes['text-decoration']:
                ids_to_be_underlined.append(css_id)
    return ids_to_be_underlined

def get_row_col(cell_ids):
    """
    Extracts row and column numbers from a list of cell IDs.

    Args:
        cell_ids: A list of cell IDs in the format "T_1f8db_row{row}_col{col}".

    Returns:
        A list of tuples, where each tuple contains (row, col) as integers. 
    """

    row_col_list = []
    for cell_id in cell_ids:
        match = re.match(r'T_\w+_row(\d+)_col(\d+)', cell_id)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            row_col_list.append((row, col))
    return row_col_list

def underline_cell(latex_table, row_col_pairs):
    """
    Underlines numeric values in specific cells of a LaTeX table. 
    Skips the first cell in each row, preserves line endings, and 
    removes content between \caption and \label (or table/tabular),
    while keeping \caption and \label.

    Args:
        latex_table: LaTeX table code as a string.
        row_col_pairs: List of (row, col) tuples for cells to underline.

    Returns:
        Modified LaTeX table code as a string.
    """

    table_content = re.search(r'\\midrule(.*?)\\bottomrule', latex_table, re.DOTALL).group(1).strip()
    table_lines = table_content.split(r'\\')

    # Create a list to store the modified table lines
    modified_lines = []

    for line_num, line in enumerate(table_lines):
        # Skip the first line (header)
        if line_num == 0: 
            modified_lines.append(line)
            continue

        # Split the line into cells
        cells = line.split('&')

        # Check if this line and any cell in it needs underlining
        for col_num, cell in enumerate(cells):
            if (line_num, col_num-1) in row_col_pairs:
                # Find and underline the number in the cell
                cell = re.sub(r'(\d+(\.\d*)?|\.\d+)', r'\\underline{\1}', cell)
            cells[col_num] = cell

        # Join the modified cells back into a line
        modified_lines.append('&'.join(cells))

    # Construct the modified table 
    modified_table = latex_table.replace(table_content, '\\\\\n'.join(modified_lines))
    return modified_table

def clean_latex_table(latex_table):
    """Cleans a LaTeX table by removing unwanted content between 
       \begin{table} and \begin{tabular}, while preserving \caption 
       and \label commands.
    """

    lines = latex_table.splitlines()
    output_lines = []
    in_removal_zone = False

    for line in lines:
        line = line.strip()

        if line.startswith(r'\begin{table}'):
            in_removal_zone = True
            output_lines.append(line)

        elif line.startswith(r'\caption'):
            output_lines.append(line)

        elif line.startswith(r'\label'):
            output_lines.append(line)

        elif line.startswith(r'\begin{tabular}'):
            in_removal_zone = False
            output_lines.append(line)

        elif not in_removal_zone:
            output_lines.append(line)

    return '\n'.join(output_lines)

def underline_table(html_table, latex_table):
    """
    Underlines specific cells in a LaTeX table based on the CSS styles of an HTML table.
    This function parses the CSS from the provided HTML table to determine which cells should be underlined.
    It then identifies the corresponding cells in the LaTeX table and underlines them. Additionally, it cleans
    the LaTeX table and adds a \resizebox command to ensure the table fits within the page width.
    Args:
        html_table (str): The HTML table as a string, containing the CSS styles.
        latex_table (str): The LaTeX table as a string.
    Returns:
        str: The modified LaTeX table with the specified cells underlined and formatted.
    """

    # Parse the CSS from the HTML table into a dictionary
    css_dict = parse_css(html_table)
    
    # Get the IDs that should be underlined
    ids_to_be_underlined = get_ids_to_undeline(css_dict)

    # Get the row and column numbers from the cell IDs
    row_col_list_to_underline = get_row_col(ids_to_be_underlined)

    # Underline the cells in the LaTeX table
    underlined_latex = underline_cell(latex_table, row_col_list_to_underline)

    # Remove unnecessary text
    underlined_latex = clean_latex_table(underlined_latex)

    # Add \resizebox before \begin{tabular}
    underlined_latex = underlined_latex.replace(
    r'\begin{tabular', 
    r'\resizebox{1\linewidth}{!}{' + '\n' + r'\begin{tabular'
)
    underlined_latex = underlined_latex.replace(
        r'\end{tabular}',
        r'\end{tabular}'+ '\n' +'}'
    )


    return underlined_latex

def dataframe_to_latex(stylized_df, filename, label, caption):
    """
    Converts a stylized pandas DataFrame to a LaTeX table and saves it to a file.
    Parameters:
    stylized_df (pandas.io.formats.style.Styler): The stylized DataFrame to be converted.
    filename (str): The name of the file where the LaTeX table will be saved.
    label (str): The label for the LaTeX table, used for referencing in LaTeX documents.
    caption (str): The caption for the LaTeX table.
    Returns:
    None
    """

    # Get the HTML and Latex tables from the styled DataFrame
    html_table = stylized_df.to_html(convert_css=True, hrules=True)
    latex_table = stylized_df.to_latex(convert_css=True, hrules=True, label=label, caption=caption)

    # Underline the cells in the LaTeX table
    underlined_latex = underline_table(html_table, latex_table)

    # Save the underlined LaTeX table to a file
    with open(filename, 'w') as f:
        f.write(underlined_latex)

    return

def convert_pd_with_mean_std_string_or_float_to_float(s):
  """
  Converts a string or float to a float.
  If the input is a string, it is assumed to be in the format 'mean ± std'.
  """
  if isinstance(s, str):
    return float(s.split(' ± ')[0])
  return s

def round_std_and_mean_or_float(s, decimal):
    """
    Rounds a string or float to a given number of decimal places.
    If the input is a string, it is assumed to be in the format 'mean ± std'.
    """
    if isinstance(s, str):
      if '±' in s:
        mean, std = s.split(' ± ')
        return f'{float(mean):.{decimal}f} ± {float(std):.{decimal}f}'
      else:
        return f'{float(s):.{decimal}f}'
    return f'{s:.{decimal}f}'

def get_cells_colors(s, threshold=5):
    s = convert_pd_with_mean_std_string_or_float_to_float(s)
    
    if s >= threshold:
        return 'background-color: gray'
    else:
        return 'background-color: white'

def highlight_table(df, color_df):
  """
  Highlights the best and second best values in each row of a pandas DataFrame.
  Applies styles for white background, bold best values, underlined second best,
  and gray background for values above 5. Robustly handles string values.

  Args:
    df (pd.DataFrame): The DataFrame to be styled.
    color_df (pd.DataFrame): The DataFrame that will be used to color the cells.

  Returns:
    pd.io.formats.style.Styler: The styled DataFrame.
  """

  counter_rows = [0]

  def apply_styles(s, color_df, counter_rows):
    # Convert the Series to floats
    s = s.apply(convert_pd_with_mean_std_string_or_float_to_float).values
    
    # Get available idxs
    available_idxs = set(np.arange(len(s)))

    # Remove idxs where color_df.iloc[counter_rows[0]] is gray
    if color_df is not None:
      gray_idxs = np.where(color_df.iloc[counter_rows[0]].apply(lambda x: 'gray' in x))[0]
      if len(gray_idxs)>0:
        available_idxs -= set(gray_idxs)
    
    # Get the best and second best indices
    if len(available_idxs)>0:
      best = np.min([s[i] for i in available_idxs])
      bests_idxs = np.where(s == best)[0]
      available_idxs -= set(bests_idxs)
    else:
      bests_idxs = []
    if len(available_idxs)>0:
      second_best = np.min([s[i] for i in available_idxs])
      second_best_idxs = np.where(s == second_best)[0]
    else:
      second_best_idxs = []
    
    # Style best and second best values
    styles = ['']*len(s)
    for best_idx in bests_idxs:
      styles[best_idx] += 'font-weight: bold;'
    for second_best_idx in second_best_idxs:
      styles[second_best_idx] += 'text-decoration: underline;'

    counter_rows[0] = counter_rows[0] + 1
    return styles
  
  # Convert values to 2 decimal places, considering that could be strings in the format 'mean ± std'
  df = df.applymap(lambda x: round_std_and_mean_or_float(x, 2))


  # Apply the style to the entire DataFrame with a white background
  headers = {
    'selector': 'th.col_heading',
    'props': 'background-color: white; color: black; font-weight: normal; border: 1px solid black'
  }
  indexes = {
    'selector': 'th:not(.index_name)',
    'props': 'background-color: white; color: black; font-weight: normal; border-right: 1px solid black; border-left: 1px solid black'
  }
  values = {
    'selector': 'td',
    'props': 'color: black; border-left: 1px solid black; border-right: 1px solid black !important'
  }
  styled_df = df.style.apply(lambda x: color_df.applymap(get_cells_colors), axis=None)
  styled_df = styled_df.set_table_styles([headers, indexes, values]).apply(apply_styles, axis=1, color_df=color_df.applymap(get_cells_colors), counter_rows=counter_rows)


  return styled_df