from src.score.cmmd import compute_cmmd

dataset_path = 'py-CPDM1.1/src/results/train/cifar100CI_init_task+5tasks_rndseed=20/CPDM/alexnetCI_pretrained_imgnet/gridsearch/demo/dm=0.5_df=0.5_e=100_bs=256_diffusion_steps=1000_num_samples=20_ddpm=100_nearest_label=cosine_tau=0.0001_image_condition=learn_image_condition_learn_lr=0.01_image_condition_learn_L2=0.01_max_grad_norm_L2=0.0005_ulhyper=3.0_classifier_type=self/task_2/TASK_TRAINING/samples'
samples_path = dataset_path
cmmd = compute_cmmd(dataset_path, samples_path, batch_size=50, max_count=-1)

print(cmmd)