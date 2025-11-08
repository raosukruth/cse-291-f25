# cse-291-f25

Must have data from https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university on your device

1. pip3 install -r requirements.txt
2. python3 models/pull.py      # Uncomment SD-3 code if disk space is available
3. python3 src/process_reports.py # Uncomment SD-3 code if disk space is available

If you want to change the number of files to run on     
  images = get_images_list(images_dir, number_of_images=<num_images_to_run_on>)
in process_report.py

Output "report.csv" will be stored in outputs/
