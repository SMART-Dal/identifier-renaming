import requests
import os
from tqdm import tqdm
import csv
import argparse
import zipfile
import shutil
import glob

def download_repo(repo_url, output_dir):
    # Extract repo name from URL
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    
    # Try 'main' branch first, then 'master' if 'main' fails
    for branch in ['main', 'master']:
        zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
        
        # Send a GET request to the ZIP file URL
        response = requests.get(zip_url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Get the total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Construct the output file path
            zip_path = os.path.join(output_dir, f"{repo_name}.zip")
            
            # Open the output file and write the content
            with open(zip_path, 'wb') as file, tqdm(
                desc=f"Downloading {repo_name} ({branch} branch)",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            
            print(f"Downloaded: {zip_path}")
            
            # Unzip the file
            repo_dir = unzip_repo(zip_path, output_dir, repo_name)
            
            # Remove the ZIP file
            os.remove(zip_path)
            print(f"Removed ZIP file: {zip_path}")
            
            # Copy Java files to root of output directory
            copy_java_files(repo_dir, output_dir)
            
            # Delete the unzipped repository folder
            shutil.rmtree(repo_dir)
            print(f"Deleted unzipped folder: {repo_dir}")
            
            # Successfully downloaded and processed, so we can break the loop
            break
    else:
        # This else clause is executed if the loop completes without breaking
        print(f"Failed to download {repo_url} (tried both 'main' and 'master' branches)")

def unzip_repo(zip_path, output_dir, repo_name):
    print(f"Unzipping {repo_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the name of the top-level directory in the ZIP file
        top_level_dir = zip_ref.namelist()[0].split('/')[0]
        
        # Extract all files
        zip_ref.extractall(output_dir)
        
        # Rename the extracted directory to match the repo name
        extracted_path = os.path.join(output_dir, top_level_dir)
        final_path = os.path.join(output_dir, repo_name)
        shutil.move(extracted_path, final_path)
    
    print(f"Unzipped to: {final_path}")
    return final_path

def copy_java_files(src_dir, dest_dir):
    print(f"Copying Java files from {src_dir} to {dest_dir}...")
    # Use glob to find all .java files in the source directory and its subdirectories
    java_files = glob.glob(os.path.join(src_dir, '**', '*.java'), recursive=True)
    
    for java_file in tqdm(java_files, desc="Copying files", unit="files"):
        # Get the base name of the file (without path)
        file_name = os.path.basename(java_file)
        # Construct the destination path
        dest_path = os.path.join(dest_dir, file_name)
        
        # If a file with the same name already exists, add a suffix
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(file_name)
            dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        # Copy the file
        shutil.copy2(java_file, dest_path)

def read_repos_from_csv(csv_file):
    repo_urls = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if 'Github Link' in row:
                repo_urls.append(row['Github Link'])
    return repo_urls

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download GitHub repositories, copy Java files to output directory, and remove temporary files and folders.")
    parser.add_argument("--csv_file", help="Path to the CSV file containing repository URLs")
    parser.add_argument("--output_dir", help="Directory to save the Java files")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read repository URLs from the CSV file
    repo_urls = read_repos_from_csv(args.csv_file)
    
    # Download, unzip, copy Java files, and clean up each repository
    for repo_url in repo_urls:
        download_repo(repo_url, args.output_dir)

if __name__ == "__main__":
    main()