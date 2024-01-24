import os
from datetime import datetime
import pandas as pd
import argparse


def process_file(file_path, output_file):
    print(f"Processing {file_path}")
    # log file does not exist, create it
    log = pd.DataFrame(columns=['timestamp', 'where', 'message'])
    summary = pd.DataFrame(columns=['type', 'value', 'generation'])
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.split(" - ")
            if len(columns) >= 3 and columns[2] in ['INFO', 'ERROR', 'WARNING']:
                timestamp = columns[0]
                where = columns[1]
                # flatten the rest of the columns as the message
                message = " - ".join(columns[3:])
                log.loc[len(log)] = [datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f'), where, message]

    assert len(log) > 0, f"Log {file_path} cannot be processed, summary is empty"

    # Drop any duplicate rows, if any - keep the first one make sure they have the same timestamp, where and message
    log = log.drop_duplicates(subset=['timestamp', 'where', 'message'], keep='first')

    start_time = log.iloc[0]['timestamp']
    end_time = log.iloc[-1]['timestamp']

    generation = None
    generation_start = None
    for row in log.itertuples():
        if row.where == "GlulamPackagingProc":
            if row.message.startswith("First feasible solution found after"):
                # extract the minutes from the message, by grepping for the number between "after " and " minutes"
                duration = float(row.message.split("after ")[1].split(" minutes")[0])
                summary.loc[len(summary)] = ['first_feasible', duration * 60, generation]
            if row.message.startswith("Presses in use: "):
                presses = int(row.message.split("Presses in use: ")[1].split(" out of ")[0])
                summary.loc[len(summary)] = ['presses', presses, generation]
                kmax = int(row.message.split(" out of ")[1])
                if presses < kmax:
                    summary.loc[len(summary)] = ['presses_not_full', kmax, generation]
            if row.message.startswith("Total waste: "):
                waste = float(row.message.split("Total waste: ")[1].split(" m^2")[0])
                summary.loc[len(summary)] = ['waste', waste, generation]
            if row.message.startswith("Total patterns: "):
                patterns = int(row.message.split("Total patterns: n=")[1])
                summary.loc[len(summary)] = ['total_patterns', patterns, generation]
        elif row.where == "GlulamES":
            if row.message.startswith("Initialising the Evolutionary Search"):
                generation_start = row.timestamp
                generation = 0
                # get all rows that have generation None
                not_remove = summary['generation'].isnull()
                if not all(not_remove):
                    print(f"Warning: removed {sum(not_remove == False)} rows")
                    summary = summary.drop(summary[summary['generation'] >= 0].index)
            if row.message.startswith("Generation"):
                generation = int(row.message.split(" ")[1].split("/")[0])
                generation_start = row.timestamp
            elif row.message.startswith("Stats - Generation"):
                duration = (row.timestamp - generation_start).total_seconds()
                summary.loc[len(summary)] = ['generation', duration, generation]
        elif row.where == "GlulamPatternProcessor":
            pass
        elif row.where == "ExtendedGlulamPatternProcessor":
            pass

    total_time = (end_time - start_time).total_seconds()
    # append to summary
    summary.loc[len(summary)] = ['total_runtime', total_time, generation]

    # save summary to log file
    summary.to_csv(output_file, index=False)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glulam Production Optimizer")
    parser.add_argument(
        "--raw_log", type=str, required=True,
        help="Path to the raw log file (default: %(default)s)"
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Path to the output file (default: %(default)s)"
    )
    args = parser.parse_args()
    # make sure the file exists
    assert os.path.exists(args.raw_log), f"File {args.raw_log} does not exist"

    process_file(args.raw_log, args.output_file)
