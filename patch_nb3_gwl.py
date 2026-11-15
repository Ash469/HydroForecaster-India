import nbformat

path = "notebooks/03_PanIndia_Data_Cleaning_Pipeline.ipynb"
nb = nbformat.read(path, as_version=4)

c11 = nb.cells[11].source
target = """            # 1. Apply Range Truncation Filter
            df_filt = df[(df[spec["col"]] >= spec["vmin"]) & (df[spec["col"]] <= spec["vmax"])].copy()"""

replacement = """            # 1. Apply Range Truncation Filter (Standardize positive m bgl to negative m bgl)
            if spec["cat"] == "gwl":
                if df[spec["col"]].median() > 0:
                    df[spec["col"]] = -df[spec["col"]].abs()
            df_filt = df[(df[spec["col"]] >= spec["vmin"]) & (df[spec["col"]] <= spec["vmax"])].copy()"""

if target in c11:
    nb.cells[11].source = c11.replace(target, replacement)
    nbformat.write(nb, path)
    print("SUCCESS: Updated Notebook 03 with Assam GWL sign handler!")
else:
    print("ERROR: Target string not found in Cell 11.")

