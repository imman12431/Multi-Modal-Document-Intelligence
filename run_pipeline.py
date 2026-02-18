import subprocess
import sys
import os


# --------------------------------------------------
# Helper: run a pipeline step
# --------------------------------------------------

def run_command(command, description):

    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        print("\n❌ Step failed!")
        return False

    print("\n✅ Step completed")
    return True


# --------------------------------------------------
# Main pipeline runner
# --------------------------------------------------

def main():

    print("\n🚀 Multi-Modal RAG Pipeline\n")

    python_exec = sys.executable  # ensures venv python is used

    steps = [

        ([python_exec, "config.py"],
         "STEP 0 — Create directories"),

        ([python_exec, "process_document.py"],
         "STEP 1 — Extract multimodal document data"),

        ([python_exec, "create_embeddings.py"],
         "STEP 2 — Generate Titan embeddings"),

        ([python_exec, "vector_store.py"],
         "STEP 3 — Build FAISS index"),
    ]

    for command, description in steps:

        if not run_command(command, description):

            print("\n💥 Pipeline stopped.")
            sys.exit(1)

    print("\n🎉 PIPELINE COMPLETE — Ready for retrieval + QA\n")


# --------------------------------------------------

if __name__ == "__main__":
    main()
