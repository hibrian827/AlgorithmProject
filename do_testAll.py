import os
import time
import subprocess

# Path to the folder containing test files
test_folder = "tests"
res = {}

# Ensure the folder exists
if not os.path.isdir(test_folder):
  print(f"Folder '{test_folder}' not found.")
  exit(1)

# Iterate over all files in the folder
for filename in sorted(os.listdir(test_folder)):
  filepath = os.path.join(test_folder, filename)
  res[filename] = {
    "valid" : False,
    "cost" : -1,
    "time" : 0
  }
  
  # Skip if it's not a regular file
  if not os.path.isfile(filepath):
    continue

  start_time = time.time()
  try:
    print(f"[*] testing {filename}")
    result = subprocess.run(
      ["./tsp", filepath],
      capture_output=True,
      text=True,
      check=True
    )
    stdout = result.stdout.strip().split(" ")
    end_time = time.time()

    if len(stdout) > 1 :
      res[filename]["valid"] = (stdout[1] == stdout[-1])
      res[filename]["cost"] = stdout[0]
    res[filename]["time"] = end_time - start_time

  except subprocess.CalledProcessError as e:
    print(f"Error running './tsp {filename}': {e}")
    print(e.stderr)

for filename in sorted(os.listdir(test_folder)):
  print(filename.ljust(30, " "), "|", str(res[filename]["valid"]).ljust(5, " "), str(res[filename]["cost"]).rjust(5, " "), str(round(res[filename]["time"] * 1000)).rjust(4, " "))