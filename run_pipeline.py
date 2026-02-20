import subprocess
import sys


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
         "STEP 1 — Extract text, tables, and images from PDF"),

        ([python_exec, "create_embeddings.py"],
         "STEP 2 — Summarize images/tables via Nova, embed summaries via Titan"),

    ]

    for command, description in steps:

        if not run_command(command, description):
            print("\n💥 Pipeline stopped.")
            sys.exit(1)

    print("\n🎉 PIPELINE COMPLETE — embedded_items.json ready.")
    print("   Launch the app with: streamlit run app.py\n")


# --------------------------------------------------

if __name__ == "__main__":
    main()