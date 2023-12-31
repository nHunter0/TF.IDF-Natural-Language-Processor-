import os
import tkinter as tk
import pandas as pd
from io import StringIO
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog
from tfidf_calculator import read_file, calculate_tfidf  # importing the function from tfidf_calculator.py
from sklearn.feature_extraction.text import TfidfVectorizer

# Dictionary to store file paths and their corresponding PDF contents
doc_contents = {}
vectorizer = TfidfVectorizer(stop_words='english')

def process_files(filepaths):
    laoding_label = tk.Label(root, text="Loading...")
    laoding_label.grid(row=0,column=0)
    root.update_idletasks() # force GUI update (due to tk having 1 thread)

    for filepath in filepaths:
        try:
            listbox.insert(tk.END, filepath)
            
            # Check if the file is a PDF or TXT
            if os.path.isfile(filepath) and os.path.splitext(filepath)[1] in ['.pdf', '.txt']:
                print(f"File path: {filepath}")
                file_content = read_file(filepath)  # Get the file content
                if file_content is not None:
                    doc_contents[filepath] = file_content  # Store the file content in the dictionary
                else:
                    tbox.insert(tk.END, f"{filepath} is not a valid pdf or txt file.")
                    print(f"Error {filepath} is not a valid pdf or txt file.")
            else:
                tbox.insert(tk.END, f"{filepath} is not a file.")
                print(f"Error {filepath} is not a file.")
        except Exception as e:
             tbox.insert(tk.END, f"Failed to process {filepath}. Reason: {str(e)}")
             print(f"Failed to process {filepath}. Reason: {str(e)}")
    laoding_label.grid_forget() # done loading

# Function to handle file drop
def drop(event):
    filepaths = event.data.split("\n") # not working with multiple drops
    process_files(filepaths)

def on_select(event):
    # Get the currently selected item in the listbox
    selection = event.widget.curselection()
    
    if selection:
        index = selection[0]
        filepath = event.widget.get(index)
        if filepath in doc_contents:
            file_content = doc_contents[filepath]  # Get the previously computed PDF content from the dictionary
            tbox.delete('1.0', tk.END)  # Clear the text box
            tbox.insert(tk.END, file_content)  # Insert the PDF content into the text box

def calculate_tfidf(corpus):
    # Calculate TF-IDF for the entire corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df

def calculate_tfidf_button_click():
    # Get the currently selected item in the listbox
    selection = listbox.curselection()
    if selection:
        index = selection[0]
        filepath = listbox.get(index)
        
        if filepath in doc_contents:
            try:
                corpus = list(doc_contents.values())  # Get the entire set of documents' content
                tfidf_df = calculate_tfidf(corpus)  # Get the TF-IDF dataframe for the entire corpus

                # Get the TF-IDF for the selected document
                tfidf_output = tfidf_df.loc[index].to_string()

                tbox.delete('1.0', tk.END)  # Clear the text box
                tbox.insert(tk.END, tfidf_output)  # Insert the TF-IDF output into the text box
            except Exception as e:
                tbox.delete('1.0', tk.END)
                tbox.insert(tk.END, f"Failed to process {filepath}. Reason: {str(e)}")
                print(f"Failed to process {filepath}. Reason: {str(e)}")
        else:
            tbox.delete('1.0', tk.END)  
            tbox.insert(tk.END, "ERROR: Selected file is not a valid")  

def open_file_dialog():
    filetypes = [("PDF files", "*.pdf"), ("Text files", "*.txt")]
    filepaths = filedialog.askopenfilenames(filetypes=filetypes, multiple=True)
    process_files(filepaths)

def download_csv():
    # Get the currently selected item in the listbox
    selection = listbox.curselection()
    if selection:
        index = selection[0]
        filepath = listbox.get(index)
        
        if filepath in doc_contents:
            file_content = doc_contents[filepath]  # Get the previously computed PDF content from the dictionary
            tfidf_output = calculate_tfidf(file_content)  # Get the TF-IDF output

            # Convert back to dataframe after to_String
            df = pd.read_csv(StringIO(tfidf_output), sep="\s+")

            # Save file locally
            save_filepath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV files", "*.csv")])
            if save_filepath:
                df.to_csv(save_filepath, index=False)
                print(f"{save_filepath} was sucessfully saved")
            else:
                print("Error occured when saving file")
        else:
            tbox.delete('1.0', tk.END) 
            tbox.insert(tk.END, "ERROR: Selected file is not a valid.") 


root = TkinterDnD.Tk()
root.geometry("900x500")
root.title("TFIDF Calculator")

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
