import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    ##する変数
    seed = cfg.seed #シード値
    tuningName = "tuning2"#tensorBoardでの識別用
    epochs = 3  # 学習回数(エポック)を指定
    ##----------------

    print(cfg.model)


if __name__ == '__main__':
    main()
