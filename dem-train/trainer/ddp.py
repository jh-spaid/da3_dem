"""--ddp
--train_roots=/gcs/dem_002/val
--out_dir=/gcs/dem_002/runs/l4x2_ddp
--model_name=depth-anything/DA3-BASE
--use_lora
--lora_r=4
--lora_alpha=16
--lora_dropout=0.0
--batch_size=2
--epochs=60
--lr=2e-05
--lr_lora=5e-05
--wd=0.01
--amp
--seed=42
--num_workers=4
--target_size=518
--dem_clamp_min=-200
--dem_clamp_max=2000
--ndsm_clamp_max=300
--lambda_ndsm=0.0
--grad_w_dem=0.5
--grad_w_ndsm=0.0
--huber_w_dem=10.0
--huber_w_ndsm=0.0
--huber_delta_dem=10.0
--huber_delta_ndsm=1.0
--viz_samples=3
--init_from_stats
--stats_max_files=200
--stats_max_vals=2000000
--use_wandb
--wandb_project=da3-dem-ndsm 
--wandb_run_name=v3_feats_lora_exp01
--resume=gs://dem_002/runs/l4x2_ddp/best.pt
"""

#g2-standard-24


gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="da3-dem-l4x2-ddp" \
  --worker-pool-spec=replica-count=1,machine-type=g2-standard-24,accelerator-type=NVIDIA_L4,accelerator-count=2,container-image-uri="${IMAGE_URI}",env=WANDB_API_KEY="${WANDB_API_KEY}",env=WANDB_PROJECT="${WANDB_PROJECT}" \
  --args="--ddp,--train_roots=/gcs/dem_002/val,--out_dir=/gcs/dem_002/runs/l4x2_ddp2,--model_name=depth-anything/DA3-BASE,--use_lora,--lora_r=4,--lora_alpha=16,--lora_dropout=0.0,--batch_size=2,--epochs=60,--lr=5e-06,--lr_lora=2e-05,--wd=0.02,--amp,--seed=42,--num_workers=4,--target_size=518,--dem_clamp_min=-200,--dem_clamp_max=2000,--ndsm_clamp_max=300,--lambda_ndsm=0.0,--grad_w_dem=0.02,--grad_w_ndsm=0.0,--huber_w_dem=10.0,--huber_w_ndsm=0.0,--huber_delta_dem=10.0,--huber_delta_ndsm=1.0,--viz_samples=3,--init_from_stats,--stats_max_files=200,--stats_max_vals=2000000,--use_wandb,--wandb_project=da3-dem-ndsm,--wandb_run_name=v3_feats_lora_exp01,--resume=/gcs/dem_002/runs/l4x2_ddp/best.pt"