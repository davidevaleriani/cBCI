# Collaborative Brain-Computer Interfaces (cBCI) Toolbox
This repository contains Python analytical tools and libraries to study the decision-making performance of groups using different approaches to integrate individual responses, including standard majority and weighted majority based on confidence estimates provided by the user or decoded by the BCI from the neural activity.

Main contributor: [Davide Valeriani](https://www.davidevaleriani.it)

# How to install

To clone and run this application, you'll need [Git](https://git-scm.com) and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer. From your Anaconda prompt command line:

```bash
# Clone this repository
git clone https://github.com/davidevaleriani/cBCI.git

# Create a new conda environment
conda create --name bci python=3.9.6

# Activate environment
conda activate bci

# Go into the repository
cd cBCI

# Install using pip
pip install .
```

# Examples and Tutorials

The folder [examples/](examples) contains sample scripts on how to use the package. Full documentation is still under development.

Examples of studies employing this package and techniques are:
- Valeriani *et al.* (2022). Multimodal collaborative brain-computer interfaces aid human-machine team decision-making in a pandemic scenario. [Journal of Neural Engineering](https://doi.org/10.1088/1741-2552/ac96a5). 
- Salvatore, Valeriani, Piccialli, Bianchi (2022). Optimized Collaborative Brain-Computer Interfaces for Enhancing Face Recognition. [IEEE Transactions on Neural Systems and Rehabilitation Engineering](https://doi.org/10.1109/TNSRE.2022.3173079).
- Valeriani, Poli (2019). Cyborg groups enhance face recognition in crowded environments. [PLOS ONE](https://doi.org/10.1371/journal.pone.0212935).
- Valeriani, Cinel, Poli (2017). Group Augmentation in Realistic Visual-Search Decisions via a Hybrid Brain-Computer Interface. [Scientific Reports](http://dx.doi.org/10.1038/s41598-017-08265-7).
- Valeriani, Poli, Cinel (2016). Enhancement of Group Perception via a Collaborative Brain-Computer Interface. [IEEE Transactions on Biomedical Engineering](http://dx.doi.org/10.1109/TBME.2016.2598875).
- Poli, Valeriani, Cinel (2014). Collaborative Brain-Computer Interface for Aiding Decision-making. [PLOS ONE](http://dx.doi.org/10.1371/journal.pone.0102693).

# Citation

The package is still under development and an associated publication will follow. In the meantime, you can cite:

> Valeriani, D., O'Flynn, L. C., Worthley, A., Hamzehei Sichani, A., & Simonyan, K. (2022). Multimodal collaborative brain-computer interfaces aid human-machine team decision-making in a pandemic scenario. Journal of Neural Engineering.

# License

Subject to [GNU GPL v3.0 license](LICENSE).