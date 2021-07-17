python -m rl.main_latent_robot --env_name Box-aside-v0 --test_env_name Box-aside-v0 --seed 829 --n_cycles 15 --clip_inputs --normalize_inputs --gamma 0.99 --n_initial_rollouts 100 --n_test_rollouts 10 --plan_eps 0.5 --n_latent_landmarks 80 --latent_batch_size 150 --n_extra_landmark 20 --dist_clip -15.0 --start_planning_n_traj 6000 --use_forward_empty_step --cuda
python -m rl.main_latent_robot --env_name Box-aside-v0 --test_env_name Box-aside-v0 --seed 252 --n_cycles 15 --clip_inputs --normalize_inputs --gamma 0.99 --n_initial_rollouts 100 --n_test_rollouts 10 --plan_eps 0.5 --n_latent_landmarks 80 --latent_batch_size 150 --n_extra_landmark 20
python -m rl.main_latent_robot --env_name Box-aside-v0 --test_env_name Box-aside-v0 --seed 173 --n_cycles 15 --clip_inputs --normalize_inputs --gamma 0.99 --n_initial_rollouts 100 --n_test_rollouts 10 --plan_eps 0.5 --n_latent_landmarks 80 --latent_batch_size 150 --n_extra_landmark 20