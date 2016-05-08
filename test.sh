./bin/singa-run.sh -exec examples/dpm/dpm.bin -conf examples/dpm/job_nuh_stepwise_train_10_every_30000.conf 2>&1 | tee output_train_file_10_time_updater_stepwise_every_30000_steps
./bin/singa-run.sh -exec examples/dpm/dpm.bin -conf examples/dpm/job_nuh_stepwise_every_1000.conf 2>&1 | tee output_train_file_1_time_updater_stepwise_every_1000_steps
./bin/singa-run.sh -exec examples/dpm/dpm.bin -conf examples/dpm/job_nuh_stepwise_every_3000.conf 2>&1 | tee output_train_file_1_time_updater_stepwise_every_3000_steps
./bin/singa-run.sh -exec examples/dpm/dpm.bin -conf examples/dpm/job_nuh_rmsprop.conf 2>&1 | tee output_train_file_10_time_updater_rmsprop
