from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Assignment 4: CI Pipeline Bug Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

pdf = PDF()
pdf.add_page()

pdf.chapter_title("1. Indentation Issues")
pdf.chapter_body(
    "Bug: The YAML file lacked proper indentation for essential blocks such as 'on', 'jobs', 'steps', and the array elements beneath them. In YAML, indentation is strictly required to define structure.\n"
    "Solution: Added 2-space indentation correctly across all hierarchical levels. e.g., indenting 'push', 'pull_request', 'branches', 'validate-and-test', and the 'steps' list."
)

pdf.chapter_title("2. Missing Checkout Step")
pdf.chapter_body(
    "Bug: The pipeline attempts to set up Python and install dependencies without explicitly checking out the project code from the repository.\n"
    "Solution: Added the 'actions/checkout@v4' step at the very beginning of the job to ensure the workflow has access to 'requirements.txt' and the Python codebase."
)

pdf.chapter_title("3. Linter Check Missing Command")
pdf.chapter_body(
    "Bug: The '- name: Linter Check' step had no 'run' or 'uses' command attached to it, which would result in a syntax error when parsed by GitHub Actions.\n"
    "Solution: Added a 'run' block to install flake8 and run it across the codebase with specific error selections (E9, F63, F7, F82)."
)

pdf.chapter_title("4. Incorrect Library in Model Dry Test")
pdf.chapter_body(
    "Bug: The 'Model Dry Test' attempts to import 'torch', but the project's dependencies ('requirements.txt') use TensorFlow and Keras.\n"
    "Solution: Changed the import statement inside the 'python -c' command from 'import torch' to 'import tensorflow', which correctly validates the environment based on the actual requirements."
)

pdf.chapter_title("5. 'on' Trigger Modification")
pdf.chapter_body(
    "Task: The 'on' trigger needed to run on every push for all branches EXCEPT 'main'.\n"
    "Solution: Updated the trigger using 'branches-ignore:\n  - main' under the 'push' event."
)

pdf.chapter_title("6. Upload README.md as Artifact")
pdf.chapter_body(
    "Task: A final step was required to upload the project's README.md as an artifact named 'project-doc'.\n"
    "Solution: Added the 'actions/upload-artifact@v4' action, setting the name parameter to 'project-doc' and the path to 'README.md'."
)

pdf.output('assignment4_bug_report.pdf')
print("PDF created successfully.")
