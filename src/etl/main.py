'''
Usage
To run the ETL pipeline with logging of each script's output and a summary at the end:

python main.py --log-output --summary
To run the ETL pipeline without logging each script's output but still get a summary at the end:

python main.py --summary
To run the ETL pipeline without logging each script's output and without a summary:

python main.py
This setup provides flexibility in how much information is logged and ensures a comprehensive overview of the ETL process
'''



import subprocess
import logging
import time
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_script(script_name, log_output):
    start_time = time.time()
    try:
        logging.info(f"Running script: {script_name}")
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Completed script: {script_name} in {elapsed_time:.2f} seconds")
        
        if log_output:
            logging.info(result.stdout)
            if result.stderr:
                logging.error(result.stderr)
                
        return elapsed_time, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script: {script_name}\n{e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline runner")
    parser.add_argument("--log-output", action="store_true", help="Log output of each script")
    parser.add_argument("--summary", action="store_true", help="Print summary at the end")
    args = parser.parse_args()
    
    scripts = [
        "extract/extract_climavi_tree_data.py",
        "extract/extract_openmeteo_weather_data.py",
        "extract/extract_climavi_weather_data.py",
        "extract/extract_climavi_forecast_data.py",
        "transform/merge_tree_openmeteo_weather_hourly_data.py",
        #"transform/merge_climavi_sensors_hourly_data.py",
        "transform/create_data_reports.py",
        "transform/create_plots.py",
        "transform/detect_irrigations.py",
    ]

    total_elapsed_time = 0
    results = []

    try:
        for script in scripts:
            elapsed_time, stdout, stderr = run_script(script, args.log_output)
            total_elapsed_time += elapsed_time
            results.append({
                'script': script,
                'elapsed_time': elapsed_time,
                'stdout': stdout,
                'stderr': stderr
            })

        logging.info(f"ETL pipeline completed successfully in {total_elapsed_time:.2f} seconds.")
        
        if args.summary:
            print("\nETL Pipeline Summary:")
            for result in results:
                print(f"Script: {result['script']}")
                print(f"Elapsed Time: {result['elapsed_time']:.2f} seconds")
                print("Output:")
                print(result['stdout'])
                if result['stderr']:
                    print("Errors:")
                    print(result['stderr'])
                print("-" * 40)
                
    except Exception as e:
        logging.error(f"ETL pipeline failed: {e}")
