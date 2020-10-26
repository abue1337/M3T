# MetaBYOL
This repo contains a new approach for representation learning based on the concepts of BYOL(Bootstrap your own latent) and MAML(Model agnostic meta learning).

# Results for MetaByol training


| Run                     | Inner steps | inner BS | outer BS | inner LR | outer LR | Train/Val Acc |
| :----------------------:|:-----------:|:--------:|:--------:|:--------:|:--------:|:-------------:|
| run_2020-10-22T17-38-25 | 1           | 8        | 4        | 0.1      | 0.1      |  0.91 / 0.859 |
| run_2020-10-23T08-11-38 | 3           | 8        | 4        | 0.1      | 0.1      |   canceled    |
| run_2020-10-23T08-56-48 | 3           | 8        | 4        | 0.01     | 0.1      | 0.989 / 0.887 |
| run_2020-10-23T08-30-43 | 1           | 8        | 32       | 0.1      | 0.1      | 0.871 / 0.832 |
| run_2020-10-23T06-40-55 | 1           | 16       | 16       | 0.1      | 0.1      | 0.912 /0.866  |
