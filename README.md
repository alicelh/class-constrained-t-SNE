# class-constrained t-SNE
Codes for our paper "[Class-Constrained t-SNE: Combining Data Features and Class Probabilities](https://ieeexplore.ieee.org/document/10294259)". This is a dimensionality reduction-based method named class-constrained t-SNE which enables the combination and comparison of data feature structure and class probability structure. 
|![projection results of a synthetic dataset](demoshowcase.png)|
|:--:| 
| *projection results of a synthetic dataset* |

|![projection results of a synthetic dataset](fashionshowcase.png)|
|:--:| 
| *projection results of the fashion MNIST dataset* |
## Usage
### How to run examples

python and conda is installed
```
cd examples
conda create --prefix ./envs notebook matplotlib numpy pandas seaborn
conda activate ./envs
jupyter notebook
```
`test.ipynb` uses the C++ version implementation and requires an executable file compliled on your own computer to run. 
`test_python_version.ipynb` uses the python version implementation which can be easily run on your computer.

### How to use the method in your project
&#x1F603; A new python version is implemented which can be found in `/examples/cstsne_python
/cstsne.py`. This version can be easily imported in your own project. 

&#x1F61E; The original version is implemented using C++, which is further wrapped using pybind11 and compiled into an executable file(.pyd in Windows). This allows for direct importation into Python for test. However, the executable code needs to be compiled on your system to function correctly, as it varies across different operating systems and architectures. Instructions for compiling the code using pybind11 can be found at https://pybind11.readthedocs.io/en/latest/index.html. After you get the executable file, you can refer the examples to import it and use it in your own project.

## Citation

If you use this code for your research, please consider citing:
```
@ARTICLE{10294259,
  author={Meng, Linhao and van den Elzen, Stef and Pezzotti, Nicola and Vilanova, Anna},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Class-Constrained t-SNE: Combining Data Features and Class Probabilities}, 
  year={2024},
  volume={30},
  number={1},
  pages={164-174},
  doi={10.1109/TVCG.2023.3326600}}
```

## Acknowledgments
Our code is build upon [HDI](https://github.com/Nicola17/High-Dimensional-Inspector) and [tSNE](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_t_sne.py).

