from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
import argparse  
import wandb

def get_args():
    parser = argparse.ArgumentParser(description='Multilayer Feedforward Neural Network')

    parser.add_argument('--wandb_project', type=str, default='video_diffusion', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='jay_gupta-indian-institute-of-technology-madras', help='WandB entity name')
    parser.add_argument('--dataset', type=str, default='vorticity over cylinder', help='dataset name')


    return parser.parse_args()



args = get_args()


# wandb.init(project=args.wandb_project,
#             entity=args.wandb_entity,
#             config=args,
#             name=f"conditional_vorticity_experiment_1")



model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 10,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    './flow_dataset', 
    './mask_dataset',                     
    train_batch_size =4,
    train_lr = 1e-4,
    save_and_sample_every = 100,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder = './results_vorticity_conditioned',               
    num_sample_rows=2  
)

trainer.train()