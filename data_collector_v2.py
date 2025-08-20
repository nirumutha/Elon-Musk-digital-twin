import os

CORPUS_DIR = "./corpus/"

def verify_data():
    """Verifies that the necessary data files exist in the corpus directory."""
    print("--- Verifying Data Collection ---")
    
    required_files = [
        "youtube_ted_talk_2017.txt",
        "blog_wait_but_why_neuralink.txt"
    ]
    
    all_files_found = True
    if not os.path.exists(CORPUS_DIR):
        print(f"🔴 ERROR: Corpus directory '{CORPUS_DIR}' not found.")
        all_files_found = False
    else:
        for filename in required_files:
            file_path = os.path.join(CORPUS_DIR, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"   ✅ Found and verified: {filename}")
            else:
                print(f"   🔴 ERROR: File not found or is empty: {filename}")
                all_files_found = False

    if all_files_found:
        print("\n--- Data Verification Complete ---")
    else:
        print("\n--- Data Verification Failed. Please create the missing files. ---")

if __name__ == "__main__":
    verify_data()
