import os
import streamlit as st
import subprocess
from datetime import datetime

def upload_data(uploaded_files):
    if not uploaded_files:
        st.warning("Please upload files first.")
        return None

    # Generate a timestamp for the folder name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    folder_name = f"data-{timestamp}"

    # Create a folder for the uploaded files
    folder_path = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder_path)

    # Save uploaded files to the folder
    for file in uploaded_files:
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

    return {"name": folder_name, "path": folder_path}

def finetune(dataset):
    folder_path = dataset["path"]

    # Ensure the folder path is a valid directory
    if os.path.isdir(folder_path):
        st.success(f"Fine-tuning on dataset: {dataset['name']}")

        # Command to run the finetune.py script
        command = ["python", "finetune.py", '--data', folder_path]

        try:
            # Run the command
            subprocess.run(command, check=True)
            st.success("Fine-tuning completed successfully!")

        except subprocess.CalledProcessError as e:
            st.error(f"Error during fine-tuning: {e}")

    else:
        st.error("Please select a valid folder.")

def get_existing_datasets():
    datasets = []
    for item in os.listdir(os.getcwd()):
        if os.path.isdir(item) and item.startswith("data-"):
            datasets.append({"name": item, "path": os.path.join(os.getcwd(), item)})
    return datasets

def main():
    st.title("Finetune Clip")

    # Get existing datasets
    datasets = st.session_state.get("datasets", get_existing_datasets())

    # "New Dataset" button
    if st.button("New Dataset"):
        uploaded_files = st.file_uploader("Drag and drop files here", type=["png", "jpg", "jpeg", 'txt'], accept_multiple_files=True)

        if uploaded_files:
            new_dataset = upload_data(uploaded_files)
            if new_dataset:
                datasets.append(new_dataset)
                st.success(f"Dataset '{new_dataset['name']}' created!")

    # Display the table of datasets
    if datasets:
        table_data = [{"Dataset Name": dataset["name"], "Fine-tune": dataset["name"]} for dataset in datasets]
        selected_dataset = st.table(table_data)

        # Check if any "Fine-tune" button is clicked
        for dataset in datasets:
            if selected_dataset.button(dataset["name"]):
                finetune(dataset)

        # Save datasets to session state
        st.session_state.datasets = datasets

    else:
        st.warning("No datasets available. Please create a new dataset.")

if __name__ == "__main__":
    main()
