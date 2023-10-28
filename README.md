This is a fork of the [`OpenOOD`](https://github.com/Jingkang50/OpenOOD) repository, containing cadet_intra and cadet_outer postprocessors from the paper [`CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning`](https://arxiv.org/abs/2210.01742).

The code to compute the intra similarity score is in `postprocessors/cadet_intra_postprocessors.py` and the code to compute the outer similarity score is in `postprocessors/cadet_outer_postprocessors.py` (Note: the outer similarity postprocessor is not implemented yet and will be in the upcoming days).

# Citation

please cite our paper as:

> @inproceedings{guille2023cadet, <br>
>  title={CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning}, <br>
>  author={Guille-Escuret, Charles and Rodriguez, Pau and Vazquez, David and Mitliagkas, Ioannis and Monteiro, Joao}, <br>
>  booktitle={Advances in Neural Information Processing Systems}, <br>
>  year={2023} <br>
>}