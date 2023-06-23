import os
import tkinter as tk
import pandas as pd
from io import StringIO
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
from tfidf_calculator import read_pdf, calculate_tfidf  # importing the function from tfidf_calculator.py

# Dictionary to store file paths and their corresponding PDF contents
pdf_contents = {}

# Function to handle file drop
def drop(event):
    filepath = event.data
    process_file(filepath)

def process_file(filepath):
    listbox.insert(tk.END, filepath)
    
    # Check if the file is a PDF
    if os.path.isfile(filepath) and os.path.splitext(filepath)[1] == '.pdf':
        print(f"File path: {filepath}")
        pdf_content = read_pdf(filepath)  # Get the PDF content
        pdf_contents[filepath] = pdf_content  # Store the PDF content in the dictionary
    else:
        tbox.insert(tk.END, f"{filepath} is not a pdf.")
        print(f"Error {filepath} is not a file.")

def on_select(event):
    # Get the currently selected item in the listbox
    selection = event.widget.curselection()
    
    if selection:
        index = selection[0]
        filepath = event.widget.get(index)
        if filepath in pdf_contents:
            pdf_content = pdf_contents[filepath]  # Get the previously computed PDF content from the dictionary
            tbox.delete('1.0', tk.END)  # Clear the text box
            tbox.insert(tk.END, pdf_content)  # Insert the PDF content into the text box

def calculate_tfidf_button_click():
    # Get the currently selected item in the listbox
    selection = listbox.curselection()
    if selection:
        index = selection[0]
        filepath = listbox.get(index)
        
        # Check if the selected file is a valid PDF
        if filepath in pdf_contents:
            pdf_content = pdf_contents[filepath]  # Get the previously computed PDF content from the dictionary
            tfidf_output = calculate_tfidf(pdf_content)  # Get the TF-IDF output
            tbox.delete('1.0', tk.END)  # Clear the text box
            tbox.insert(tk.END, tfidf_output)  # Insert the TF-IDF output into the text box
        else:
            tbox.delete('1.0', tk.END)  # Clear the text box
            tbox.insert(tk.END, "ERROR: Selected file is not a valid PDF.")  # Show error message

def open_file_dialog():
    filepath = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    process_file(filepath)

def download_csv():
    # Get the currently selected item in the listbox
    selection = listbox.curselection()
    if selection:
        index = selection[0]
        filepath = listbox.get(index)
        
        # Check if the selected file is a valid PDF
        if filepath in pdf_contents:
            pdf_content = pdf_contents[filepath]  # Get the previously computed PDF content from the dictionary
            tfidf_output = calculate_tfidf(pdf_content)  # Get the TF-IDF output

            # Convert back to dataframe after to_String
            df = pd.read_csv(pd.compat.StringIO(tfidf_output), sep="\s+")

            # Save file locally
            save_filepath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV files", "*.csv")])
            if save_filepath:
                df.to_csv(save_filepath, index=False)
                print(f"{save_filepath} was sucessfully saved")
            else:
                print("Error occured when saving file")
        else:
            tbox.delete('1.0', tk.END)  # Clear the text box
            tbox.insert(tk.END, "ERROR: Selected file is not a valid PDF.")  # Show error message

root = TkinterDnD.Tk()
root.geometry("900x500")

# Listbox Scroll Bar 
lb_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
lb_scrollbar.grid(row=1, column=0,  padx=10, pady=1, sticky='ew')

# Listbox
listbox = tk.Listbox(root, selectmode=tk.SINGLE, background="#ffe0d6", xscrollcommand=lb_scrollbar.set)
listbox.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky='nsew')
listbox.drop_target_register(DND_FILES)
listbox.dnd_bind("<<Drop>>", drop)
listbox.bind('<<ListboxSelect>>', on_select)  # Bind the function to the listbox select event

# Textbox Scroll Bar 
tb_scrollbar = tk.Scrollbar(root)
tb_scrollbar.grid(row=0, column=3, padx=10, pady=10, sticky='ns')

# Text box
tbox = tk.Text(root, yscrollcommand=tb_scrollbar.set)
tbox.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

# Calculate TF-IDF Button
calculate_tfidf_button = tk.Button(root, text='Calculate TF-IDF', command=calculate_tfidf_button_click)
calculate_tfidf_button.grid(row=2, column=2, padx=10, pady=10, sticky='w')

# Open file dialog button
open_file_button = tk.Button(root, text='Open File', command=open_file_dialog)
open_file_button.grid(row=2, column=0, padx=10, pady=10, sticky='w')

# Download CSV button
download_csv_button = tk.Button(root, text='Download CSV', command=download_csv)
download_csv_button.grid(row=2, column=2, padx=10, pady=10, sticky='e')

# Configure column weight to expand the horizontal scrollbar
root.grid_columnconfigure(0, weight=1)

# Configurations
tb_scrollbar.config(command=tbox.yview)
lb_scrollbar.config(command=listbox.xview)

root.mainloop()
