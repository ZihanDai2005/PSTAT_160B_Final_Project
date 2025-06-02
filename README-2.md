## Run the Code Locally (First Time Setup)

1. Open a terminal and navigate to the project root directory
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

---

## Scramble Text

To generate a scrambled version of the original text:
`python3 code/scramble_text.py -i original.txt > secret_message.txt`

---

## Execution Order

0. Each time you work on the project, activate the virtual environment: `source venv/bin/activate`
1. utils.py
2. deciphering_utils.py
3. metropolis_hastings.py
4. run_deciphering.py
   `python3 code/run_deciphering.py -i data/warpeace_input.txt -d secret_message.txt`
