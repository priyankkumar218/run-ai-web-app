import os
import streamlit as st
import subprocess

def finetune(folder_path):
    # Ensure the folder path is a valid directory
    if os.path.isdir(folder_path):
        st.success(f"Fine-tuning on folder: {folder_path}")

        # Command to run the finetune.py script
        command = ["python", "finetune.py", folder_path]

        try:
            # Run the command
            subprocess.run(command, check=True)
            st.success("Fine-tuning completed successfully!")

        except subprocess.CalledProcessError as e:
            st.error(f"Error during fine-tuning: {e}")

    else:
        st.error("Please select a valid folder.")

def main():
    st.title("Fine-tune App")

    # Allow the user to drag and drop a folder
    folder_path = st.file_uploader("Drag and drop a folder here", type="folder")

    if folder_path:
        # Fine-tune button
        if st.button("Fine-tune"):
            finetune(folder_path)

if __name__ == "__main__":
    main()
