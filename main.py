import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Image FileName: {cfg.img_filename}")


if __name__ == "__main__":
    main()
