## DIGITS EXPERIMENTS

To download the datasets, you can e.g. unpack the following [file](https://github.com/thuml/CDAN/blob/master/data/usps2mnist/images.tar.gz) in data/digits.
For the sake of speed when running experiments, the code generates a pkl file containing the whole datasets (when first run) and then loads it at runtime.

To run the code on the original datasets, run:

`python train_digits.py {method} --task {task}`

where method belongs to: [`CDAN`, `CDAN-E`, `DANN`, `IWDAN`, `NANN`, `IWDANORACLE`, `IWCDAN`, `IWCDANORACLE`, `IWCDAN-E`, `IWCDAN-EORACLE`] and task is either mnist2usps or usps2mnist.

To run the code on the subsampled datasets, run:

`python train_digits.py {method} --task {task} --ratio 1`

To reproduce Figs. 1 and 2, run the following command with various seeds:

`python train_digits.py {method} --ratio {ratio}`

where ratio belongs to 100 <= ratio < 150 (to subsample the target) or to 200 <= ratio < 250 (to subsample the source). Each corresponds to a given subsampling, the exact fractions can be found in the subsampling list of `data_list.txt.

## VISDA AND OFFICE DATASETS

The Visda dataset can be found here: https://github.com/VisionLearningGroup/taskcv-2017-public.

The Office-31 dataset can be found here: https://people.eecs.berkeley.edu/~jhoffman/domainadapt.

The Office-Home dataset can be found here: http://hemanthdv.org/OfficeHome-Dataset.

They should be downloaded and placed in the corresponding folder under data. The code will generate a test file once for faster evaluation, it might take a while during the first visda run.

### Discriminator based methods

To run the code on the original datasets, run:

`python train_image.py {method} --dset {dset} --s_dset_file {s_dset_file} --t_dset_file {t_dset_file}`

where:
  - `method` belongs to [`CDAN`, `CDAN-E`, `DANN`, `IWDAN`, `NANN`, `IWDANORACLE`, `IWCDAN`, `IWCDANORACLE`, `IWCDAN-E`, `IWCDAN-EORACLE`]
  - `dset` belongs to [`visda`, `office-31`, `office-home`]
  - `s_dset_file` corresponds to the source domain, the filename can be found in the corresponding data folder, e.g. `dslr_list.txt` (not needed for VISDA)
  - `t_dset_file` corresponds to the target domain, the filename can be found in the corresponding data folder, e.g. `amazon_list.txt` (not needed for VISDA).

To run the code on the subsampled datasets, run the same command with `--ratio 1` appended to it.

### MMD-based methods

To run the MMD algorithms (e.g. IWJAN), use the same commands as above with the `train_mmd.py` file.

## Reference

Please consider citing us if you use this code:

```
@inproceedings{tachet2020domain,
      title={Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift},
      author={Tachet des Combes, Remi and Zhao, Han and Wang, Yu-Xiang and Gordon, Geoff},
      year={2020},
      booktitle={Advances in Neural Information Processing Systems}
}
```

This code is based upon the work of:

```
@inproceedings{long2018conditional,
  title={Conditional adversarial domain adaptation},
  author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1645--1655},
  year={2018}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.