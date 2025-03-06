from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess

def run_script(script_path):
	with open(script_path, 'r') as file:
		exec(file.read())


with DAG(
	'sequential_scripts_dag',
	default_args={
		'owner': 'airflow', 
		'start_date': datetime.now(),
		'retries': 1,
		'retry_delay': timedelta(minutes=5)
	},
	catchup=False
) as dag:

	split = 'held_out'

	impute_cost = PythonOperator(
		task_id='impute_cost',
		python_callable=run_script,
		op_args=['impute_cost.py', split]
	)

	pick_prediction_point = PythonOperator(
		task_id='pick_prediction_point',
		python_callable=run_script,
		op_args=['pick_prediction_point.py', split]
	)


	map_cohort = PythonOperator(
		task_id='map_halo_cohort_for_finetune',
		python_callable=run_script,
		op_args=['map_halo_cohort_for_finetune.py', split]
	)

	impute_cost >> pick_prediction_point >> map_cohort

